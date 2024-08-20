# encoding: utf-8
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1 # must match ./yolox/data/datasets/coco_classes.py file
        self.warmup_epochs = 1
        self.max_epoch = 500
        
        # choose depth & width based on model used
        # self.depth = 0.33 # YOLOX-s
        # self.width = 0.50 # YOLOX-s
        # self.depth = 0.67 # YOLOX-m
        # self.width = 0.75 # YOLOX-m
        self.depth = 1.0 # YOLOX-l
        self.width = 1.0 # YOLOX-l
        # self.depth = 1.33 # YOLOX-x
        # self.width = 1.25 # YOLOX-x
        
        self.print_interval = 1
        self.eval_interval = 1 # epoch interval for evaluation on validation set
        self.save_history_ckpt = False # save all trained ckpt histories
         
        self.data_dir = "./datasets/glove" # data directory here
        self.train_ann = "train.json"
        self.val_ann = "valid.json"
        self.test_ann = "test.json" # not used; can ignore if want

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.output_dir = '/mnt/data/glove/models/v2' # update output base directory; a new training folder will be created here

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
                cache=cache,
                cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="valid",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )