from typing import Any, List, Tuple, Callable
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
from PIL import Image
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.callback.wandb import *
import torch
import torchvision
import wandb
from fastaiMetrics import acc_segmentation, MIOU
import gc
wandb.login()

class KFoldTraining():
    
    def __init__(self, sprintName:str, dataset_path:str, fnames:List, get_y_fn: Callable, codes:List, kFolds: int, X_test: List = None,
                test_size: float = 0.2, shuffle: bool = True, useRandomSeed: bool = True, isGoogleColab: bool = True) -> None:
        
        self.sprint = sprintName
        self.path = dataset_path
        self.fnames = fnames
        self.get_y_fn = get_y_fn
        self.codes = codes
        self.isGoogleColab = isGoogleColab
        
        randomSeed = None
        if useRandomSeed:
            randomSeed = 2         

        if X_test is None:
            self.X_train, self.X_test, _, _ = train_test_split(self.fnames, [0] * len(self.fnames), test_size=test_size,
                                                                random_state=randomSeed, shuffle=True, stratify=None)
        else:
            self.X_train = [file for file in fnames if file not in X_test]
            self.X_test = X_test
            
        print("Training and Validation dataset size:", len(self.X_train))
        print("Test dataset size:", len(self.X_test))

        # set kfold splits
        kf = KFold(n_splits=kFolds, shuffle=shuffle, random_state=randomSeed)
        self.kfSplits = kf.split(self.X_train)
       
    def train(self, bs:int=8, gradientAcc:int=2, lr:float = 5e-4, freezeEpochs:int = 20, unfreezeEpochs:int = 20, wd:float = 1e-2, backbone = resnet34) -> Tuple[List,List]:

        validation_metrics = []
        test_metrics = []

        # train k splits and evaluate each of them
        for splitIdx, (train_idxs, val_idxs) in enumerate(self.kfSplits):

            # configure dataloaders and learner
            dataloaders = self._createTrainValidationDataloaders(val_idxs, bs) 
            test_dl = self._getTestDataloader(dataloaders)
            learner = self._createUnetLearner(dataloaders, backbone)
            print("Length of trainingset",len(dataloaders.train_ds))
            print("Length of validationset",len(dataloaders.valid_ds))
            print("Length of testset",len(test_dl.dataset))

            # run this kfold split and track experiment with wandb
            runName = self.sprint + "-kFoldSplit-"+str(splitIdx) + "-randomId-" + str(np.random.randint(1024))
            run = wandb.init(project="diabetesLearningKfold", name=runName) # track machine learning experiment
            fitCallbacks = [WandbCallback(log='all'), SaveModelCallback(every_epoch=False, monitor='miou', fname='bestModel-'+runName,with_opt=True), GradientAccumulation(n_acc=gradientAcc)]

            learner.fine_tune(unfreezeEpochs, base_lr=lr, freeze_epochs=freezeEpochs, pct_start=0.3, wd=wd, cbs=fitCallbacks)
            validation_metrics.append(learner.validate())
            test_metrics.append(learner.validate(dl=test_dl))
            run.finish()
            
            # # clean memory before going to next run
            del learner
            del dataloaders
            del test_dl
            torch.cuda.empty_cache()
            gc.collect()

        return validation_metrics, test_metrics


    def _createMetrics(self) -> List[Any]:
        classes_index = range(1, len(self.codes)) #exclude background class at index 0
        metrics = [acc_segmentation, DiceMulti, MIOU(classes_index, axis=1)]
        return metrics
        
    def _createTrainValidationDataloaders(self, val_idxs:List[int], bs:int):

        size = (256, 256)
        item_tfms = [Resize(size, method=ResizeMethod.Squish, resamples=(Image.NEAREST,Image.NEAREST))]
        aug_tfms = aug_transforms(mult=1, flip_vert=True, size=size)

        datablock = DataBlock(blocks=(ImageBlock, MaskBlock(self.codes)),
                        splitter=IndexSplitter(val_idxs),
                        get_y=self.get_y_fn,
                        item_tfms=item_tfms, 
                        batch_tfms=[*aug_tfms, Normalize.from_stats(*imagenet_stats)])

        dataloaders = datablock.dataloaders(self.X_train, path=self.path, bs=bs)
        return dataloaders

    def _getTestDataloader(self, dataloaders):
        test_dl = dataloaders.test_dl(self.X_test, with_labels=True)
        test_dl.vocab = self.codes
        return test_dl


    def _createUnetLearner(self, dataloaders, backbone) -> Learner:
        metrics = self._createMetrics()
        modelCallbacks = [ShowGraphCallback]
        opt_func = Adam
        loss_func = FocalLossFlat(weight=None, axis=1)

        learner = unet_learner(dataloaders, backbone, loss_func=loss_func, opt_func=opt_func, metrics=metrics, cbs=modelCallbacks,
                               self_attention=False, act_cls=Mish).to_fp32()

        return learner

