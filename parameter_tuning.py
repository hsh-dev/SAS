from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager
from Models.SAS_Model import SAS
from config import config

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


def experiment(experiment_dir, hparams):
    with tf.summary.create_file_writer(experiment_dir).as_default():
        hp.hparams(hparams)
        score = run_model(hparams)
        tf.summary.scalar(METRIC_HR, score, step=1)


def run_model(hparams):
    new_config = config
    new_config['hidden_dim'] = hparams[HP_HID_UNIT]
    new_config['sequence_length'] = hparams[HP_SEQ_LEN]
    new_config['attention_layer_count'] = hparams[HP_ATT_UNIT]
    new_config['optimizer'] = hparams[HP_OPTIMIZER]
    new_config['max_epoch'] = 3
    
    dataloader = DataLoader(new_config)
    item_count = dataloader.get_movie_len()

    model = SAS(i_dim=item_count,
                d_dim=new_config["hidden_dim"],
                n_dim=new_config["sequence_length"],
                b_num=new_config["attention_layer_count"])

    trainmanger = TrainManager(model, dataloader, new_config)
    best_score = trainmanger.start()
    
    return best_score


if __name__ == "__main__":    
    
    '''
    Hyperparameter Range Initialization
    '''
    HP_HID_UNIT = hp.HParam('hidden_units', hp.Discrete([30, 40, 50]))
    HP_SEQ_LEN = hp.HParam('sequence_length', hp.Discrete([5, 10, 15, 20]))
    # HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.5))
    HP_ATT_UNIT = hp.HParam('attention_units', hp.Discrete([2, 3, 4]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['ADMA', 'SGD']))

    METRIC_HR = 'hit_rate'

    with tf.summary.create_file_writer('log/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_HID_UNIT, HP_SEQ_LEN, HP_ATT_UNIT, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_HR, display_name='HitRate')],)


    experiment_no = 0
    for optimizer in HP_OPTIMIZER.domain.values:
        for hid_units in HP_HID_UNIT.domain.values:
            for seq_len in HP_SEQ_LEN.domain.values:
                for att_units in HP_ATT_UNIT.domain.values:
                    hparams = {
                        HP_HID_UNIT: hid_units,
                        HP_SEQ_LEN: seq_len,
                        HP_ATT_UNIT: att_units,
                        HP_OPTIMIZER: optimizer}

                experiment_name = f'Experiment {experiment_no}'
                print(f'Starting Experiment: {experiment_name}')
                print({h.name: hparams[h] for h in hparams})
                experiment('log/hparam_tuning/' + experiment_name, hparams)
                experiment_no += 1
