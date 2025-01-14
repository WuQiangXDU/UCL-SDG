import numpy as np
import argparse
import os
from models.ucl_sdg import UCL_SDG
from utils import datautils
from pandas import DataFrame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='CWRU', choices='PU, CWRU, JNU')
    parser.add_argument('--data_dir', type=str, default='Datasets')
    parser.add_argument('--transfer_task', type=list, default=[[0], [0, 1, 2, 3]], help='transfer learning tasks')
    parser.add_argument('--signal_size', type=int, default=1024, help='signal size')
    parser.add_argument('--num_class', type=int, default=10, help='number of categories')
    parser.add_argument('--num_dom', type=int, default=4, help='number of domains')

    parser.add_argument('--hp', type=float, default=5, help='hp')

    parser.add_argument('--small_sample_task', default=False, help='Controls whether small sample tasks are performed')
    parser.add_argument('--source_train_size', type=int, default=10, help='source_train_size')

    parser.add_argument('--epochs', type=int, default=300, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size (defaults to 64)')
    parser.add_argument('--model_name', type=str, default='Simple_ResNet', help='the name of the model')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate (defaults to 0.01)')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='exp',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110',
                        help='the learning rate decay for step and stepLR')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_threads', type=int, default=None, help='The maximum allowed number of threads used by this process')

    args = parser.parse_args()

    for args.data_name in ['CWRU', 'PU', 'JNU']:

        args.data_dir = f'Datasets/{args.data_name}'

        # Different tasks are set for different datasets in the format [[source domain code], [target domains code]]
        if args.data_name == 'CWRU':
            transfer_task_all = [[[0], [0, 1, 2, 3]], [[1], [0, 1, 2, 3]], [[2], [0, 1, 2, 3]], [[3], [0, 1, 2, 3]]]
            args.num_class = 10
            args.num_dom = 4

        elif args.data_name == 'PU':
            transfer_task_all = [[[0], [0, 1, 2, 3]], [[1], [0, 1, 2, 3]], [[2], [0, 1, 2, 3]], [[3], [0, 1, 2, 3]]]
            args.num_class = 6
            args.num_dom = 4

        elif args.data_name == 'JNU':
            transfer_task_all = [[[0], [0, 1, 2]], [[1], [0, 1, 2]], [[2], [0, 1, 2]]]
            args.num_class = 4
            args.num_dom = 3

        for args.transfer_task in transfer_task_all:

            task_name = 'S' + str((args.transfer_task[0])[0])
            acc_all = []

            save_dir = f'results/{args.data_name}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for n in range(10):   # repeat 10 times
                print(f"\n**********************************************", args.data_name, args.transfer_task, '---', n)

                # load data
                print('Loading data... ', end='')
                all_data_label = datautils.load_Cross_domain(args)
                print('done')

                model = UCL_SDG(args=args, lr=args.lr, batch_size=args.batch_size)
                loss_log, acc = model.fit(all_data_label, n_epochs=args.epochs, verbose=True)

                transfer_task_source_array = np.array((args.transfer_task[0]))
                transfer_task_target_array = np.array(args.transfer_task[1])
                acc_all.append(acc)
                acc_all_array = np.array(acc_all)

                mean_value = np.mean(acc_all_array, axis=0)
                mean_value = np.expand_dims(mean_value, axis=0)
                std_value = np.std(acc_all_array, axis=0)
                std_value = np.expand_dims(std_value, axis=0)

                nan_row = np.full((1, acc_all_array.shape[1]), np.nan)
                acc_all_array = np.concatenate((acc_all_array, nan_row, mean_value, std_value), axis=0)
                acc_all_array = np.round(acc_all_array, decimals=4)

                # save results
                index_labels = list(range(1, acc_all_array.shape[0] - 2)) + ["", "mean", "std"]
                df = DataFrame(
                    {
                        f"S{transfer_task_source_array[0]}-T{args.transfer_task[1]}": acc_all_array[:, -1],
                        **{
                            f"S{transfer_task_source_array[0]}-T{transfer_task_target_array[i]}": acc_all_array[:, i]
                            for i in range(len(transfer_task_target_array))
                        }
                    },
                    index=index_labels
                )

                df.to_excel(f'{save_dir}/{args.data_name}_{task_name}.xlsx', sheet_name='sheet1')

    print("Finished.")
