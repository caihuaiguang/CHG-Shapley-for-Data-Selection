# Learning setting
config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="mnist",
                           datadir="../data",
                           feature="dss",
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=128,
                              pin_memory=True),

              model=dict(architecture='MnistNet',
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
                             T_max=200),

              dss_args=dict(type="GLISTER",
                                fraction=0.1,
                                select_every=20,
                                kappa=0,
                                linear_layer=False,
                                selection_type='Supervised',
                                collate_fn = None,
                                greedy='Stochastic'),

              train_args=dict(num_epochs=200,
                              device="cuda",
                              print_every=20,
                              run=1,
                              results_dir='results/',
                              wandb=False,
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )