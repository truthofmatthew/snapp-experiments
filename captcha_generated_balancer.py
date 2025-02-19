import os
import glob
import random
from collections import defaultdict
from tqdm import tqdm


def get_file_list(directory, ext='*'):

    return glob.glob(os.path.join(directory, f'*.{ext}'))


def count_digits_in_filename(filename):

    label = os.path.splitext(os.path.basename(filename))[0]
    counts = defaultdict(int)
    for ch in label:
        if ch.isdigit():
            counts[ch] += 1
    return counts


def aggregate_digit_counts(file_list):

    totals = {str(d): 0 for d in range(10)}
    for f in file_list:
        for d, cnt in count_digits_in_filename(f).items():
            totals[d] += cnt
    return totals


def balance_files_by_removal(file_list, target):

    file_digit_counts = {f: count_digits_in_filename(f) for f in file_list}
    current_counts = aggregate_digit_counts(file_list)
    remaining_files = set(file_list)

    removal_possible = True
    pbar = tqdm(total=len(file_list), desc="Balancing files", leave=True)
    removed_files = []

    while removal_possible:
        removal_possible = False
        candidates = []

        for f in remaining_files:
            counts = file_digit_counts[f]
            safe = all(current_counts[d] - counts.get(d, 0) >= target for d in counts)
            if safe and any(current_counts[d] > target and counts.get(d, 0) > 0 for d in counts):
                candidates.append(f)

        if candidates:
            f_remove = random.choice(candidates)
            counts = file_digit_counts[f_remove]

            for d, cnt in counts.items():
                current_counts[d] -= cnt

            remaining_files.remove(f_remove)
            removed_files.append(f_remove)
            removal_possible = True
            pbar.update(1)

    pbar.close()
    return list(remaining_files), removed_files, current_counts


def main():
    directory = './generated_captchas'

    files = get_file_list(directory)
    print(f"Initial files: {len(files)}")

    # Count initial digit occurrences
    initial_counts = aggregate_digit_counts(files)
    print("Initial counts:", initial_counts)

    # # Define the target count per digit
    # target = 5701
    # print(f"Target count per digit: {target}")
    #
    # # Balance the files by removal
    # balanced_files, removed_files, final_counts = balance_files_by_removal(files, target)
    # print(f"Balanced files: {len(balanced_files)}")
    # print("Final counts:", final_counts)
    #
    # # Optionally, move the removed files to a separate directory
    # if removed_files:
    #     removed_dir = os.path.join(directory, 'removed')
    #     os.makedirs(removed_dir, exist_ok=True)
    #     for f in removed_files:
    #         new_path = os.path.join(removed_dir, os.path.basename(f))
    #         os.rename(f, new_path)
    #     print(f"Moved {len(removed_files)} files to '{removed_dir}'.")


if __name__ == "__main__":
    main()