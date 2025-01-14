from sklearn.model_selection import train_test_split
import numpy as np
import utils.dataset_load as dataset

def load_Cross_domain(args):

    Dataset = getattr(dataset, args.data_name)
    train_X, train_y, test_X, test_y, num_classes = Dataset(args).data_split()
    train_X, test_X_source, train_y, test_y_source = train_test_split(train_X, train_y, train_size=0.8, random_state=args.seed)
    # 'test_X_source' is the source domain dataset for testing, 'test_y_source' is the corresponding label

    for i in range(args.num_class):
        test_y_source[test_y_source == i] = (i + (args.transfer_task[0][0] * 10))

    # Small Sample Task
    if args.small_sample_task:
        for i in range(num_classes):
            x_choiced = train_X[train_y == i]
            y_choiced = train_y[train_y == i]
            assert len(y_choiced) > args.source_train_size, " 'nu_each_class' is too large"
            x_train_choiced = x_choiced[0:args.source_train_size, :, :]
            y_train_choiced = y_choiced[0:args.source_train_size]
            x_test_choiced = x_choiced[args.source_train_size:, :, :]
            y_test_choiced = y_choiced[args.source_train_size:]
            if i == 0:
                x_train = x_train_choiced
                y_train = y_train_choiced
                x_test = x_test_choiced
                y_test = y_test_choiced
                continue
            x_train = np.concatenate((x_train, x_train_choiced), axis=0)
            y_train = np.concatenate((y_train, y_train_choiced), axis=0)
            x_test = np.concatenate((x_test, x_test_choiced), axis=0)
            y_test = np.concatenate((y_test, y_test_choiced), axis=0)

    # Add the portion of the source domain data not involved in training to the test data
    test_y = np.concatenate((test_y_source, test_y), axis=0)
    test_X = np.concatenate((test_X_source, test_X), axis=0)

    return [train_X, train_y, test_X, test_y]

