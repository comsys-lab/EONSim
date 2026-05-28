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
    def build_output_dir(output_base_dir, emb_dim, vectors_per_table, num_tables, pooling_factor, batch_size, dataset_path=None):
        dataset_name = ""
        if dataset_path:
            # Create a string like dlrm_reuse_high_test from datasets/dlrm/reuse_high_test.txt
            # Remove leading directories if they are just "datasets/" for cleaner names, but robustly:
            clean_path = dataset_path.split("datasets/")[-1] if "datasets/" in dataset_path else dataset_path
            name_no_ext = os.path.splitext(clean_path)[0]
            dataset_name = "_" + name_no_ext.replace("/", "_")
        
        output_dir_name = f"{emb_dim}_{vectors_per_table}_{num_tables}_{pooling_factor}_{batch_size}{dataset_name}"
        return os.path.join(output_base_dir, output_dir_name)

    @staticmethod
    def resolve_output_dir_from_workload(output_base_dir, workload_config_base_path, batch_size, dataset_path=None):
        try:
            from .config_loader import ConfigLoader
        except ImportError:
            from config_loader import ConfigLoader

        cfg_loader = ConfigLoader(workload_config_base_path)
        emb_conf = cfg_loader.get_embedding_config()

        return Helper.build_output_dir(
            output_base_dir=output_base_dir,
            emb_dim=emb_conf['embedding_dim'],
            vectors_per_table=emb_conf['vectors_per_table'],
            num_tables=emb_conf['num_tables'],
            pooling_factor=emb_conf['pooling_factor'],
            batch_size=batch_size,
            dataset_path=dataset_path
        )


def _title_row(title, width):
    title_decorated = f"《 {title} 》"
    display_w = len(title) + 6  # 《(2cols) + ' ' + title + ' ' + 》(2cols)
    left = (width - display_w) // 2
    right = width - display_w - left
    return "║" + "▒"*left + title_decorated + "▒"*right + "║"


def print_styled_header(title):
    width = 100
    print("\n╔" + "═"*width + "╗")
    print(_title_row(title, width))
    print("╚" + "═"*width + "╝")


def print_styled_box(title, content_lines):
    width = 100
    print("\n╔" + "═"*width + "╗")
    print(_title_row(title, width))
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
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset path")
    args = parser.parse_args()

    if args.resolve_output_dir:
        if args.workload_config is None or args.output_base_dir is None or args.batch_size is None:
            parser.error("--resolve-output-dir requires --workload-config, --output-base-dir, and --batch-size")

        print(
            Helper.resolve_output_dir_from_workload(
                output_base_dir=args.output_base_dir,
                workload_config_base_path=args.workload_config,
                batch_size=args.batch_size,
                dataset_path=args.dataset_path
            )
        )


if __name__ == "__main__":
    main()
