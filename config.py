import options
# 不动
class Config(object):
    def __init__(self, args):
        self.root_dir = args.root_dir
        self.lr = eval(args.lr)
        self.num_iters = len(self.lr)
        self.batch_size = args.batch_size
        self.output_path = args.output_path
        self.seed = args.seed
        self.num_segments = args.num_segments
        self.len_feature = args.len_feature
        self.exp_name = args.exp_name
            
if __name__ == "__main__":
    args=options.parse_args()
    conf=Config(args)
    print(conf.lr)

