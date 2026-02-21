import time
import os
import argparse


class Helper:
    def init(self):
        self.start = 0
        self.end = 0

    def set_timer(self):
        self.start = time.perf_counter()

    def end_timer(self, task):
        self.end = time.perf_counter()
        print('(Time elapsed(s) in {}: {:10.6f}sec)'.format(task, self.end-self.start))

    @staticmethod
    def build_output_dir(output_base_dir, emb_dim, vectors_per_table, num_tables, pooling_factor, batch_size):
        output_dir_name = f"{emb_dim}_{vectors_per_table}_{num_tables}_{pooling_factor}_{batch_size}"
        return os.path.join(output_base_dir, output_dir_name)

    @staticmethod
    def resolve_output_dir_from_workload(output_base_dir, workload_config_base_path, batch_size):
        try:
            from .ConfigLoader import ConfigLoader
        except ImportError:
            from ConfigLoader import ConfigLoader

        cfg_loader = ConfigLoader(workload_config_base_path)
        emb_conf = cfg_loader.get_embedding_config()

        return Helper.build_output_dir(
            output_base_dir=output_base_dir,
            emb_dim=emb_conf['embedding_dim'],
            vectors_per_table=emb_conf['vectors_per_table'],
            num_tables=emb_conf['num_tables'],
            pooling_factor=emb_conf['pooling_factor'],
            batch_size=batch_size
        )


def print_styled_header(title):
    width = 100
    title_decorated = f"《 {title} 》"
    print("\n╔" + "═"*width + "╗")
    print("║" + "▒"*(width//2-len(title_decorated)//2-1) + title_decorated + "▒"*(width//2-len(title_decorated)//2-1) + "║")
    print("╚" + "═"*width + "╝")


def print_styled_box(title, content_lines):
    width = 100
    title_decorated = f"《 {title} 》"
    print("\n╔" + "═"*width + "╗")
    print("║" + "▒"*(width//2-len(title_decorated)//2-1) + title_decorated + "▒"*(width//2-len(title_decorated)//2-1) + "║")
    print("╠" + "═"*width + "╣")
    for line in content_lines:
        print("║ " + line.ljust(width-2) + " ║")
    print("╚" + "═"*width + "╝")


def main():
    parser = argparse.ArgumentParser(description="Helper utilities for EONSim")
    parser.add_argument("--resolve-output-dir", action="store_true", help="Resolve output dir from workload config")
    parser.add_argument("--workload-config", type=str, help="Workload config base path")
    parser.add_argument("--output-base-dir", type=str, help="Base output directory")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    args = parser.parse_args()

    if args.resolve_output_dir:
        if args.workload_config is None or args.output_base_dir is None or args.batch_size is None:
            parser.error("--resolve-output-dir requires --workload-config, --output-base-dir, and --batch-size")

        print(
            Helper.resolve_output_dir_from_workload(
                output_base_dir=args.output_base_dir,
                workload_config_base_path=args.workload_config,
                batch_size=args.batch_size,
            )
        )


if __name__ == "__main__":
    main()
