from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='phone-email',
                        choices=['phone-email', 'ACM-DBLP', 'foursquare-twitter'],
                        help='available datasets: phone-email, ACM-DBLP, foursquare-twitter')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.4, help='soft update parameter')
    parser.add_argument('--max_iter', dest='max_iter', type=int, default=100, help='maximum number of isorank iteration')

    return parser.parse_args()
