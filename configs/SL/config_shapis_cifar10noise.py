# Learning setting
config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="cifar10",
                           datadir="../data",
                           feature="noise",
                           noise_ratio = 0.3,
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=128,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=10),
              
              ckpt=dict(is_load=False,
                        is_save=False,
                        dir='results/',
                        save_every=20),
              
              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.05,
                             weight_decay=5e-4,
                             nesterov=True),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_args=dict(type="SHAPIS",
                            fraction=0.5, 
                            select_every = 20,
                            kappa = 0,
                            linear_layer= True,  
                            selection_type='PerClassandShap',
                            # varients = "CHGShapley",
                            varients = "GradientShapley",
                            # varients = "HardnessShapley",
                            # varients = "TracIn",
                            # varients = "Cosine",
                            p = 1, 
                            valid= False,   
                            collate_fn = None), 
                            
              train_args=dict(num_epochs=300,
                              device="cuda:4",
                              print_every=20,
                              run=1,
                              wandb=False,
                              results_dir='results/',
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )
