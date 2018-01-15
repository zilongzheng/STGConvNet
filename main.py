import argparse
import tensorflow as tf
from src.model import STGConvnet

def main():
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('-sz', '--image_size', type=int, default=224)

    # training hyper-parameters
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-num_chain', type=int, default=3, help='number of synthesized results for each batch of training data')
    parser.add_argument('-num_frames', type=int, default=70, help='number of frames used in training data')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-beta1', type=float, default=0.5, help='momentum1 in Adam')

    # langevin hyper-parameters
    parser.add_argument('-delta', '--step_size', type=float, default=0.3)
    parser.add_argument('-sample_steps', type=int, default=20)

    # misc
    parser.add_argument('-output_dir', type=str, default='./output', help='output directory')
    parser.add_argument('-category', type=str, default='fire_pot')
    parser.add_argument('-data_path', type=str,
                        default='./trainingVideo/', help='root directory of data')
    parser.add_argument('-log_step', type=int, default=10, help='number of steps to output synthesized image')

    opt = parser.parse_args()

    with tf.Session() as sess:
        model = STGConvnet(sess, opt)
        model.train()

if __name__ == '__main__':
    main()