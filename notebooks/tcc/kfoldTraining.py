from typing import Any, List, Tuple
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
    
    def __init__(self, dataset_path:str, kFolds: int, test_size: float = 0.2,
                 shuffle: bool = True, useRandomSeed: bool = True, isGoogleColab: bool = True) -> None:

        self.path = dataset_path
        self.path_img = dataset_path + 'done/'
        self.path_anno = dataset_path + 'gt/'

        self.get_y_fn = lambda x : self.path_anno + '/' + f'{x.stem}_GT.png'

        self.label_fnames = get_image_files(self.path_anno)
        self.fnames = get_image_files(self.path_img)

        self.codes = np.loadtxt(dataset_path + 'classesNumber.txt', dtype=str, delimiter='\n',encoding='utf')
        self.codes = [code.split(": ")[1] for code in self.codes] #pega apenas o nome de cada classe e ignora o ID

        self.isGoogleColab = isGoogleColab
        
        randomSeed = None
        if useRandomSeed:
            randomSeed = 2         

        # open all images to stratify based on pixel values
        # allImgs, allGTs = self._openAllFiles()

        # no stratification for now! Use balanced dataset
        self.X_train, self.X_test, _, _ = train_test_split(self.fnames, self.label_fnames, test_size=test_size,
                                                                random_state=randomSeed, shuffle=True, stratify=None)

        # set kfold splits
        kf = KFold(n_splits=kFolds, shuffle=shuffle, random_state=randomSeed)
        self.kfSplits = kf.split(self.X_train)
       
    def train(self, bs:int=8, lr:float = 2e-3, epochs:int = 20, wd:float = 1e-2) -> Tuple[List,List]:

        validation_metrics = []
        test_metrics = []

        # train k splits and evaluate each of them
        for splitIdx, (train_idxs, val_idxs) in enumerate(self.kfSplits):

            # configure dataloaders and learner
            dataloaders = self._createTrainValidationDataloaders(val_idxs, bs) 
            test_dl = self._getTestDataloader(dataloaders)
            learner = self._createUnetLearner(dataloaders)
            print("Length of training, validation and test sets")
            print(len(dataloaders.train_ds), len(dataloaders.valid_ds), len(test_dl)*bs)

            # # run this kfold split and track experiment with wandb
            runName = "kFoldSplit-"+str(splitIdx) + "-randomId-" + str(np.random.randint(1024))
            run = wandb.init(project="diabetesLearningKfold", name=runName) # track machine learning experiment
            fitCallbacks = [WandbCallback(log='all'), SaveModelCallback(every_epoch=False, monitor='acc_segmentation', fname='bestModel'+runName), GradientAccumulation(n_acc=16)]
            learner.fine_tune(epochs, base_lr=lr, freeze_epochs=3, wd=wd, cbs=fitCallbacks)
            validation_metrics.append(learner.validate())
            test_metrics.append(learner.validate(dl=test_dl))
            run.finish()
            
            # # clean memory before going to next run
            del learner
            torch.cuda.empty_cache()
            gc.collect()

        return validation_metrics, test_metrics


    def _createMetrics(self) -> List[Any]:
        classes_index = range(1, len(self.codes)) #exclude background class at index 0
        metrics = [acc_segmentation, DiceMulti, MIOU(classes_index, axis=1)]
        return metrics
        
    def _createTrainValidationDataloaders(self, val_idxs:List[int], bs:int):

        item_tfms = [Resize((512,512), method=ResizeMethod.Crop,resamples=(Image.NEAREST,Image.NEAREST))]
        aug_tfms = aug_transforms(mult=1, min_scale=1, flip_vert=True, size=(512,512), max_warp=0.2)

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


    def _createUnetLearner(self, dataloaders) -> Learner:
        metrics = self._createMetrics()
        modelCallbacks = [ShowGraphCallback,EarlyStoppingCallback(patience=20)]
        opt_func = ranger
        classWeights = torch.ones(len(self.codes)).cuda() if self.isGoogleColab else torch.ones(len(self.codes))
        classWeights[0] = 0.1
        loss_func = CrossEntropyLossFlat(weight=classWeights, axis=1)

        learner = unet_learner(dataloaders, resnet34, loss_func=loss_func, opt_func=opt_func, metrics=metrics, cbs=modelCallbacks,
                            wd_bn_bias=False, self_attention=False, act_cls=Mish)

        return learner


    def _openAllFiles(self) -> Tuple[List,List]:
        #open all gts and save them as np.array. This is necessary to stratify correctly
        allImages = []
        for idx in range(0,len(self.fnames)):
            img = Image.open(self.fnames[idx])
            arr = np.asarray(img)
            allImages.append(arr)

        #open all gts and save them as np.array. This is necessary to stratify correctly
        allGTs = []
        for idx in range(0,len(self.fnames)):
            img = Image.open(self.label_fnames[idx])
            arr = np.asarray(img)
            allGTs.append(arr)

        return allImages, allGTs