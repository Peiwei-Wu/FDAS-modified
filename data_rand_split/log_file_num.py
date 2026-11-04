from pathlib import Path


def count_files_in_folders(root_dir):
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"路径不存在: {root_dir}")

    print(f"统计根目录: {root_dir}")
    print("-" * 80)

    for folder in sorted(root_dir.rglob("*")):
        if folder.is_dir():
            # 只统计这个文件夹“当前这一层”下面的文件，不统计更深层
            files = [p for p in folder.iterdir() if p.is_file()]

            if len(files) > 0:
                rel_path = folder.relative_to(root_dir)
                print(f"{rel_path}: {len(files)} 个文件")


def main():
    root_dir = "/data/WPW/datasets/mms/"
    count_files_in_folders(root_dir)


if __name__ == "__main__":
    main()