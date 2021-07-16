# import some common libraries
import os

# import some common detectron2 utilities
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, datasets, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.engine import launch, default_argument_parser


def main(args):
    logger = setup_logger()
    # Handle COCO datasets
    register_coco_instances("coco2017_train", {}, "/home/justin/Data/Coco2017/annotations/instances_train2017.json",
                            "/home/justin/Data/Coco2017/images/train2017")
    register_coco_instances("coco2017_val", {}, "/home/justin/Data/Coco2017/annotations/instances_val2017.json",
                            "/home/justin/Data/Coco2017/images/val2017")

    # dicts = datasets.coco.load_coco_json('/home/justin/Data/Coco2017/annotations/instances_train2017.json',
    #                                      '/home/justin/Data/Coco2017/images/train2017',
    #                                      'coco2017_train')
    # logger.info("Done loading {} samples.".format(len(dicts)))
    # dirname = "coco-data-vis"
    # os.makedirs(dirname, exist_ok=True)
    # meta = MetadataCatalog.get('coco2017_train')
    # for d in dicts:
    #     img = np.array(Image.open(d["file_name"]))
    #     visualizer = Visualizer(img, metadata=meta)
    #     vis = visualizer.draw_dataset_dict(d)
    #     fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
    #     vis.save(fpath)

    # Setup config for Mask Model
    # cfg = get_cfg()
    cfg = model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", trained=False)
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("coco2017_train",)
    cfg.DATASETS.TEST = ("coco2017_val",)
    cfg.SOLVER.IMS_PER_BATCH = 15  # Number of Images per batch across ALL machines
    cfg.SOLVER.MAX_ITER = 170660  # Total number of Iterations
    cfg.SOLVER.GAMMA = 0.1
    # cfg.DATALOADER.NUM_WORKERS = 2
    # The iteration number to decrease learning rate by GAMMA.
    cfg.SOLVER.STEPS = (130000,)  # Mask decreases by 10 at 120K iteration out of 160K

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.OUTPUT_DIR = 'mask_paper_train'
    cfg.SOLVER.BASE_LR = 0.02
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]

    with open(f'/home/justin/Models/detectron2/{cfg.OUTPUT_DIR}/cfg_summary.txt', 'a+') as f:
        f.write(cfg.dump())

    # Training
    print('Training...')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluate COCO
    print('Evaluating...')
    evaluator = COCOEvaluator("coco2017_val", ("bbox", "segm"), output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "coco2017_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))

    # testing = trainer.test(cfg, trainer.model, evaluator)
    # print(testing)

    return


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # When running multi-gpu training, must be called through launch.py
    launch(main, num_gpus_per_machine=3, num_machines=1,
           dist_url=args.dist_url,
           machine_rank=0,
           args=(args,))
