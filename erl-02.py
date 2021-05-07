from parlai.scripts.display_data import DisplayData
from parlai.scripts.train_model import TrainModel


print("start training...")

TrainModel.main(
    model='seq2seq',
    model_file='model_erl01/model',
    task='dailydialog',
    max_train_time=10 * 60,
    # batchsize=1,
    # validation_every_n_secs=10,
    # max_train_time=60,
)