
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from tqdm import tqdm
import holidays

from dataset import *
from models import *
from metric_n_loss import *
    
    
def main(args):
    IS_CUSTOM = True if args.model_name=="custom" else False
    
    if args.data_from=="TURKEY":
        df = "../data/demand_df_turkey.csv"
        holidays_ = holidays.TR()
    elif args.data_from=="SPAIN":
        df = "../data/demand_df_spain.csv"
        holidays_ = holidays.ESP()
        
    #函数get_pd_dataset：读取数据、数据预处理、提取特征、数据归一化、划分数据集    
    train_df, test_df, [load_min, load_max] = get_pd_dataset(df, args.data_from, holidays_, IS_CUSTOM)
    #函数get_np_dataset：返回np类型的train_x, train_y, test_x, test_y
    train_x, train_y, test_x, test_y = get_np_dataset(train_df, test_df, IS_CUSTOM)
    #函数get_tf_dataset：返回tensor类型的train_dataset、test_dataset
    train_dataset = get_tf_dataset(train_x, train_y, args.input_n)
    test_dataset = get_tf_dataset(test_x, test_y, args.input_n)

    mape, rmse = get_unnormalized_mape(load_min, load_max), get_unnormalized_rmse(load_min, load_max)
    #训练模型
    if args.model_name == "custom":
        model = CustomModel()
    elif args.model_name == "deepenergy":
        model = DeepEnergy()
    elif args.model_name == "seqmlp":
        model = SeqMlp()
    else:
        raise ValueError("model name should be one of these 3 strings: 'custom', 'seqmlp', 'deepenergy'")
    #优化器
    optim = tf.keras.optimizers.Adam()
    model.compile(loss="mae", metrics=["mape", mape, rmse], optimizer=optim)
    #优化学习率+早停避免过拟合
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, verbose=1, restore_best_weights=True)
    #在每个training/epoch/batch结束时，可以通过回调函数Callbacks查看一些内部信息。常用的callback有EarlyStopping等
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=args.epochs, callbacks=[lr_reducer, early_stopper])

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()#创建解析器对象add_argument添加程序参数信息
    parser.add_argument("--data_from", type=str, default="SPAIN", help="Name of the country which dataset comes from")
    parser.add_argument("--model_name", type=str, default='custom', help="One of 'custom', 'seqmlp', 'deepenergy' ")
    parser.add_argument("--epochs", type=int, default=150, help="Number of times the entire dataset will be seen in training.")
    parser.add_argument("--input_n", type=int, default=24, help="Number of input time sequence. Default to 24h")
    parser.add_argument("--output_n", type=int, default=24, help="Number of output time sequence. Default to 24h")
    args = parser.parse_args()#解析参数
        
    main(args)
    
    print("Parsed arguments are", args.data_from, args.is_custom, args.input_n, args.output_n)
