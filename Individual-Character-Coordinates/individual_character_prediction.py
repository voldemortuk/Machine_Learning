import unittest
import os
import numpy as np
from PIL import Image
from tensorflow import keras

from calamari_ocr.ocr import DataSetType, PipelineParams
from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams, MultiPredictor
from calamari_ocr.utils import glob_all

from calamari_ocr.scripts.predict import run
from calamari_ocr.utils.image import load_image

this_dir = os.path.dirname(os.path.realpath(__file__))


class PredictionAttrs:
    def __init__(self):
        self.files = sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")]))
        self.checkpoint = [os.path.join(this_dir, "models", "0.ckpt")]
        self.processes = 1
        self.batch_size = 1
        self.verbose = True
        self.voter = "confidence_voter_default_ctc"
        self.output_dir = None
        self.extended_prediction_data = None
        self.extended_prediction_data_format = "json"
        self.no_progress_bars = True
        self.extension = None
        self.dataset = DataSetType.FILE
        self.text_files = None
        self.pagexml_text_index = 0
        self.beam_width = 20
        self.dictionary = []
        self.dataset_pad = None


class TestValidationTrain(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_prediction(self):
        args = PredictionAttrs()
        args.checkpoint = args.checkpoint[0:1]
        run(args)

    def test_prediction_voter(self):
        args = PredictionAttrs()
        run(args)

    def test_empty_image_raw_prediction(self):
        args = PredictionAttrs()
        predictor = Predictor.from_checkpoint(PredictorParams(progress_bar=False, silent=True), checkpoint=args.checkpoint[0])
        images = [np.zeros(shape=(0, 0)), np.zeros(shape=(1, 0)), np.zeros(shape=(0, 1))]
        for result in predictor.predict_raw(images):
            print(result.outputs.sentence)

    def test_white_image_raw_prediction(self):
        args = PredictionAttrs()
        predictor = Predictor.from_checkpoint(PredictorParams(progress_bar=False, silent=True), checkpoint=args.checkpoint[0])
        images = [np.zeros(shape=(200, 50))]
        for result in predictor.predict_raw(images):
            print(result.outputs.sentence)

    def test_raw_prediction(self):
        args = PredictionAttrs()
        predictor = Predictor.from_checkpoint(PredictorParams(progress_bar=False, silent=True), checkpoint=args.checkpoint[0])
        images = [load_image(file) for file in args.files]
        for result in predictor.predict_raw(images):
            self.assertGreater(result.outputs.avg_char_probability, 0)

    def test_raw_dataset_prediction(self):
        args = PredictionAttrs()
        predictor = Predictor.from_checkpoint(PredictorParams(progress_bar=False, silent=True), checkpoint=args.checkpoint[0])
        params = PipelineParams(
            type=DataSetType.FILE,
            files=args.files
        )
        for sample in predictor.predict(params):
            pass

    def test_raw_prediction_voted(self):
        args = PredictionAttrs()
        predictor = MultiPredictor.from_paths(checkpoints=args.checkpoint, predictor_params=PredictorParams(progress_bar=False, silent=True))
        images = [load_image(file) for file in args.files]
        for sample in predictor.predict_raw(images):
            r, voted = sample.outputs
            print([rn.sentence for rn in r])
            print(voted.sentence)

class Predictor(tfaip_cls.Predictor):
    @staticmethod
    def from_checkpoint(params: PredictorParams, checkpoint: str, auto_update_checkpoints=True):
        ckpt = SavedCalamariModel(checkpoint, auto_update=False)
        trainer_params = Scenario.trainer_params_from_dict(ckpt.dict)
        trainer_params.scenario_params.data_params.pre_processors_.run_parallel = False
        trainer_params.scenario_params.data_params.post_processors_.run_parallel = False
        scenario = Scenario(trainer_params.scenario_params)
        predictor = Predictor(params, scenario.create_data())
        ckpt = SavedCalamariModel(checkpoint, auto_update=auto_update_checkpoints)  # Device params must be specified first
        predictor.set_model(keras.models.load_model(ckpt.ckpt_path + '.h5', custom_objects=Scenario.model_cls().get_all_custom_objects()))
        return predictor


class MultiPredictor(tfaip_cls.MultiModelPredictor):
    @classmethod
    def from_paths(cls, checkpoints: List[str],
                   auto_update_checkpoints=True,
                   predictor_params: PredictorParams = None,
                   voter_params: VoterParams = None,
                   **kwargs
                   ) -> 'aip_predict.MultiModelPredictor':
        if not checkpoints:
            raise Exception("No checkpoints provided.")

        if predictor_params is None:
            predictor_params = PredictorParams(silent=True, progress_bar=True)

        DeviceConfig(predictor_params.device_params)
        checkpoints = [SavedCalamariModel(ckpt, auto_update=auto_update_checkpoints) for ckpt in checkpoints]
        multi_predictor = super(MultiPredictor, cls).from_paths(
            [ckpt.json_path for ckpt in checkpoints],
            predictor_params,
            Scenario,
            model_paths=[ckpt.ckpt_path + '.h5' for ckpt in checkpoints],
            predictor_args={'voter_params': voter_params},
        )

        return multi_predictor

    def __init__(self, voter_params, *args, **kwargs):
        super(MultiPredictor, self).__init__(*args, **kwargs)
        self.voter_params = voter_params or VoterParams()

    def create_voter(self, data_params: 'DataParams') -> MultiModelVoter:
        # Cut non text processors (first two)
        post_proc = [Data.data_processor_factory().create_sequence(
            data.params().post_processors_.sample_processors[2:], data.params(), PipelineMode.Prediction) for
            data in self.datas]
        pre_proc = Data.data_processor_factory().create_sequence(
            self.data.params().pre_processors_.sample_processors, self.data.params(),
            PipelineMode.Prediction)
        out_to_in_transformer = OutputToInputTransformer(pre_proc)
        return CalamariMultiModelVoter(self.voter_params, self.datas, post_proc, out_to_in_transformer)


@dataclass_json
@dataclass
class PredictionCharacter:
    char: str = ''
    label: int = 0
    probability: float = 0

    def __post_init__(self):
        self.probability = float(self.probability)
        self.label = int(self.label)


@dataclass_json
@dataclass
class PredictionPosition:
    chars: List[PredictionCharacter] = field(default_factory=list)
    local_start: int = 0
    local_end: int = 0
    global_start: int = 0
    global_end: int = 0


@dataclass_json
@dataclass
class Prediction:
    id: str = ''
    sentence: str = ''
    labels: List[int] = field(default_factory=list)
    positions: List[PredictionPosition] = field(default_factory=list)
    logits: Optional[np.array] = field(default=None)
    total_probability: float = 0
    avg_char_probability: float = 0
    is_voted_result: bool = False
    line_path: str = ''
    voter_predictions: Optional[List['Prediction']] = None


@dataclass_json
@dataclass
class Predictions:
    predictions: List[Prediction] = field(default_factory=list)
    line_path: str = ''


@dataclass_json
@dataclass
class PredictorParams(tfaip.PredictorParams):
    with_gt: bool = False
    ctc_decoder_params = None
    silent: bool = True


class PredictionResult:
    def __init__(self, prediction, codec, text_postproc, out_to_in_trans: Callable[[int], int], ground_truth=None):
        """ The output of a networks prediction (PredictionProto) with additional information

        It stores all required information for decoding (`codec`) and interpreting the output.

        Parameters
        ----------
        prediction : PredictionProto
            prediction the DNN
        codec : Codec
            codec required to decode the `prediction`
        text_postproc : TextPostprocessor
            text processor to apply to the decodec `prediction` to receive the actual prediction sentence
        """
        self.prediction = prediction
        self.logits = prediction.logits
        self.codec = codec
        self.text_postproc = text_postproc
        self.chars = codec.decode(prediction.labels)
        self.sentence = self.text_postproc.apply(Sample(inputs='', outputs="".join(self.chars))).outputs
        self.prediction.sentence = self.sentence
        self.out_to_in_trans = out_to_in_trans
        self.ground_truth = ground_truth

        self.prediction.avg_char_probability = 0

        for p in self.prediction.positions:
            for c in p.chars:
                c.char = codec.code2char[c.label]

            p.global_start = int(self.out_to_in_trans(p.local_start))
            p.global_end = int(self.out_to_in_trans(p.local_end))
            if len(p.chars) > 0:
                self.prediction.avg_char_probability += p.chars[0].probability

        self.prediction.avg_char_probability /= len(self.prediction.positions) if len(
            self.prediction.positions) > 0 else 1

if __name__ == "__main__":
    unittest.main()