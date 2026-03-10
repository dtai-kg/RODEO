import os
import pprint

from torch.utils.data import DataLoader, RandomSampler
from utils.data_utils import load, load_args_from_parser

args = load_args_from_parser()

if args.encoding == 'llm':

    if 'CTA-SCH' in args.tasks or 'CPA-SCH' in args.tasks:
        # LLMs experiments
        from loaders.pt_dataset_llm import (
            collate_fn,
            collate_test_fn,
            CTADataset,
            CTATestDataset,
            CPADataset,
            CPATestDataset,
        )
    else:
        raise ValueError(f"encoding='llm' is not supported for {args.tasks} tasks")

else:
    if 'CTA-GIT' in args.tasks or 'CPA-GIT' in args.tasks:
        # Metadata experiments
        from loaders.pt_dataset_metadata import (
            collate_fn,
            collate_test_fn,
            collate_rel_fn,
            CTADataset,
            CTATestDataset,
            CPADataset,
            CPATestDataset,
        )
    else:
        # Regular experiments
        from loaders.pt_dataset import (
            collate_fn,
            collate_fn_turl,
            collate_test_fn,
            CTADataset,
            CTATestDataset,
            CPADataset,
            CPATestDataset,
        )

# Regular experiments
# from loaders.pt_dataset import (
#     collate_fn,
#     collate_fn_turl,
#     collate_test_fn,
#     CTADataset,
#     CTATestDataset,
#     CPADataset,
#     CPATestDataset,
# )

# LLMs experiments
# from loaders.pt_dataset_llm import (
#     collate_fn,
#     collate_test_fn,
#     CTADataset,
#     CTATestDataset,
#     CPADataset,
#     CPATestDataset,
# )

# Metadata experiments
# from loaders.pt_dataset_metadata import (
#     collate_fn,
#     collate_test_fn,
#     collate_rel_fn,
#     CTADataset,
#     CTATestDataset,
#     CPADataset,
#     CPATestDataset,
# )

def create_loader(args):

    if 'CTA-TURL' in args.tasks or 'CPA-TURL' in args.tasks:

        dataset_path = args.turl_dataset_path
        filename1 = os.path.join(dataset_path, 'cta.pkl')
        filename2 = os.path.join(dataset_path, 'cpa.pkl')

    elif 'CTA-GIT' in args.tasks or 'CPA-GIT' in args.tasks:
        dataset_path = args.gittab_dataset_path
        filename1 = os.path.join(dataset_path, 'git.pkl')
        filename2 = filename1

    else:
        dataset_path = args.sotab_dataset_path
        if args.encoding == 'llm':
            filename1 = os.path.join(dataset_path, 'cta_llm_qwen.pkl')
            filename2 = os.path.join(dataset_path, 'cpa_llm_qwen.pkl')
            # filename1 = os.path.join(dataset_path, 'cta_llm_qwen4b.pkl')
            # filename2 = os.path.join(dataset_path, 'cpa_llm_qwen4b.pkl')
            # filename1 = os.path.join(dataset_path, 'cta_llm_qwen_tiny.pkl')
            # filename2 = os.path.join(dataset_path, 'cpa_llm_qwen_tiny.pkl')
        else:
            filename1 = os.path.join(dataset_path, 'cta.pkl')
            filename2 = os.path.join(dataset_path, 'cpa.pkl')
            # filename1 = os.path.join(dataset_path, 'cta_full.pkl')
            # filename2 = os.path.join(dataset_path, 'cpa_full.pkl')

    print (filename1)
    print (filename2)

    cta_table_dict = load(filename1)
    cpa_table_dict = load(filename2)

    data_dict = {}
    task_num_class_dict = {}
    for task in args.tasks:
        if args.encoding == 'llm':
            pkl_file = os.path.join(dataset_path, task + '_llm.pkl')
        else:
            pkl_file = os.path.join(dataset_path, task + '.pkl')
            # pkl_file = os.path.join(dataset_path, task + '_full.pkl')
        data_dict[task] = load(pkl_file)
        task_num_class_dict[task] = len(data_dict[task]['label2idx'])

    batch_size = args.batch_size

    train_datasets = []
    train_dataloaders = []
    valid_datasets = []
    valid_dataloaders = []

    for task in args.tasks:

        if task == "CTA-TURL":

            train_cta_turl_dataset = CTADataset(data_dict[task], cta_table_dict, 'train', args)
            train_cta_turl_dataset.generate_epoch()
            train_cta_turl_sampler = RandomSampler(train_cta_turl_dataset)
            train_cta_turl_dataloader = DataLoader(train_cta_turl_dataset,
                                                  sampler=train_cta_turl_sampler,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn_turl)

            valid_cta_turl_dataset = CTADataset(data_dict[task], cta_table_dict, 'dev', args)
            valid_cta_turl_dataloader = DataLoader(valid_cta_turl_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn_turl)

            train_datasets.append(train_cta_turl_dataset)
            train_dataloaders.append(train_cta_turl_dataloader)
            valid_datasets.append(valid_cta_turl_dataset)
            valid_dataloaders.append(valid_cta_turl_dataloader)

        elif task == "CPA-TURL":

            train_cpa_turl_dataset = CPADataset(data_dict[task], cpa_table_dict, 'train', args)
            train_cpa_turl_dataset.generate_epoch()
            train_cpa_turl_sampler = RandomSampler(train_cpa_turl_dataset)
            train_cpa_turl_dataloader = DataLoader(train_cpa_turl_dataset,
                                                  sampler=train_cpa_turl_sampler,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn_turl)

            valid_cpa_turl_dataset = CPADataset(data_dict[task], cpa_table_dict, 'dev', args)
            valid_cpa_turl_dataloader = DataLoader(valid_cpa_turl_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn_turl)

            train_datasets.append(train_cpa_turl_dataset)
            train_dataloaders.append(train_cpa_turl_dataloader)
            valid_datasets.append(valid_cpa_turl_dataset)
            valid_dataloaders.append(valid_cpa_turl_dataloader)


        elif task == "CTA-GIT":

            train_cta_git_dataset = CTADataset(data_dict[task], cta_table_dict, 'train', args)
            train_cta_git_dataset.generate_epoch()
            train_cta_git_sampler = RandomSampler(train_cta_git_dataset)
            train_cta_git_dataloader = DataLoader(train_cta_git_dataset,
                                                  sampler=train_cta_git_sampler,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            valid_cta_git_dataset = CTADataset(data_dict[task], cta_table_dict, 'dev', args)
            valid_cta_git_dataloader = DataLoader(valid_cta_git_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            train_datasets.append(train_cta_git_dataset)
            train_dataloaders.append(train_cta_git_dataloader)
            valid_datasets.append(valid_cta_git_dataset)
            valid_dataloaders.append(valid_cta_git_dataloader)

        elif task == "CPA-GIT":

            train_cpa_git_dataset = CPADataset(data_dict[task], cpa_table_dict, 'train', args)
            train_cpa_git_dataset.generate_epoch()
            train_cpa_git_sampler = RandomSampler(train_cpa_git_dataset)
            train_cpa_git_dataloader = DataLoader(train_cpa_git_dataset,
                                                  sampler=train_cpa_git_sampler,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_rel_fn)

            valid_cpa_git_dataset = CPADataset(data_dict[task], cpa_table_dict, 'dev', args)
            valid_cpa_git_dataloader = DataLoader(valid_cpa_git_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_rel_fn)

            train_datasets.append(train_cpa_git_dataset)
            train_dataloaders.append(train_cpa_git_dataloader)
            valid_datasets.append(valid_cpa_git_dataset)
            valid_dataloaders.append(valid_cpa_git_dataloader)

        elif task == "CTA-SCH":

            train_cta_sch_dataset = CTADataset(data_dict[task], cta_table_dict, 'train', args)
            train_cta_sch_dataset.generate_epoch()
            train_cta_sch_sampler = RandomSampler(train_cta_sch_dataset)
            train_cta_sch_dataloader = DataLoader(train_cta_sch_dataset,
                                                  sampler=train_cta_sch_sampler,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            valid_cta_sch_dataset = CTADataset(data_dict[task], cta_table_dict, 'validation', args)
            valid_cta_sch_dataloader = DataLoader(valid_cta_sch_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            train_datasets.append(train_cta_sch_dataset)
            train_dataloaders.append(train_cta_sch_dataloader)
            valid_datasets.append(valid_cta_sch_dataset)
            valid_dataloaders.append(valid_cta_sch_dataloader)

        elif task == "CPA-SCH":

            train_cpa_sch_dataset = CPADataset(data_dict[task], cpa_table_dict, 'train', args)
            train_cpa_sch_dataset.generate_epoch()
            train_cpa_sch_sampler = RandomSampler(train_cpa_sch_dataset)
            train_cpa_sch_dataloader = DataLoader(train_cpa_sch_dataset,
                                                  sampler=train_cpa_sch_sampler,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            valid_cpa_sch_dataset = CPADataset(data_dict[task], cpa_table_dict, 'validation', args)
            valid_cpa_sch_dataloader = DataLoader(valid_cpa_sch_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            train_datasets.append(train_cpa_sch_dataset)
            train_dataloaders.append(train_cpa_sch_dataloader)
            valid_datasets.append(valid_cpa_sch_dataset)
            valid_dataloaders.append(valid_cpa_sch_dataloader)

        elif task == "CTA-DBP":

            train_cta_dbp_dataset = CTADataset(data_dict[task], cta_table_dict, 'train', args)
            train_cta_dbp_dataset.generate_epoch()
            train_cta_dbp_sampler = RandomSampler(train_cta_dbp_dataset)
            train_cta_dbp_dataloader = DataLoader(train_cta_dbp_dataset,
                                                  sampler=train_cta_dbp_sampler,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            valid_cta_dbp_dataset = CTADataset(data_dict[task], cta_table_dict, 'validation', args)
            valid_cta_dbp_dataloader = DataLoader(valid_cta_dbp_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            train_datasets.append(train_cta_dbp_dataset)
            train_dataloaders.append(train_cta_dbp_dataloader)
            valid_datasets.append(valid_cta_dbp_dataset)
            valid_dataloaders.append(valid_cta_dbp_dataloader)

        elif task == "CPA-DBP":

            train_cpa_dbp_dataset = CPADataset(data_dict[task], cpa_table_dict, 'train', args)
            train_cpa_dbp_dataset.generate_epoch()
            train_cpa_dbp_sampler = RandomSampler(train_cpa_dbp_dataset)
            train_cpa_dbp_dataloader = DataLoader(train_cpa_dbp_dataset,
                                                  sampler=train_cpa_dbp_sampler,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            valid_cpa_dbp_dataset = CPADataset(data_dict[task], cpa_table_dict, 'validation', args)
            valid_cpa_dbp_dataloader = DataLoader(valid_cpa_dbp_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            train_datasets.append(train_cpa_dbp_dataset)
            train_dataloaders.append(train_cpa_dbp_dataloader)
            valid_datasets.append(valid_cpa_dbp_dataset)
            valid_dataloaders.append(valid_cpa_dbp_dataloader)

        else:
            raise ValueError("task name must be wrong.")

    for task, train_dataset in zip(args.tasks, train_datasets):
        print("{} Batch Total Num: {}".format(task, int(len(train_dataset) / batch_size)))

    return train_datasets, train_dataloaders, valid_datasets, valid_dataloaders, data_dict, task_num_class_dict

def create_test_loader(args):

    if 'CTA-TURL' in args.tasks or 'CPA-TURL' in args.tasks:
        dataset_path = args.turl_dataset_path
        filename1 = os.path.join(dataset_path, 'cta.pkl')
        filename2 = os.path.join(dataset_path, 'cpa.pkl')

    elif 'CTA-GIT' in args.tasks or 'CPA-GIT' in args.tasks:
        dataset_path = args.gittab_dataset_path
        filename1 = os.path.join(dataset_path, 'git.pkl')
        filename2 = filename1

    else:
        dataset_path = args.sotab_dataset_path
        if args.encoding == 'llm':
            filename1 = os.path.join(dataset_path, 'cta_llm_qwen.pkl')
            filename2 = os.path.join(dataset_path, 'cpa_llm_qwen.pkl')
            # filename1 = os.path.join(dataset_path, 'cta_llm_qwen4b.pkl')
            # filename2 = os.path.join(dataset_path, 'cpa_llm_qwen4b.pkl')
            # filename1 = os.path.join(dataset_path, 'cta_llm_qwen_tiny.pkl')
            # filename2 = os.path.join(dataset_path, 'cpa_llm_qwen_tiny.pkl')
        else:
            # filename1 = os.path.join(dataset_path, 'cta_full.pkl')
            # filename2 = os.path.join(dataset_path, 'cpa_full.pkl')
            filename1 = os.path.join(dataset_path, 'cta.pkl')
            filename2 = os.path.join(dataset_path, 'cpa.pkl')

    print (filename1)
    print (filename2)

    cta_table_dict = load(filename1)
    cpa_table_dict = load(filename2)

    data_dict = {}
    task_num_class_dict = {}
    for task in args.tasks:
        if args.encoding == 'llm':
            pkl_file = os.path.join(dataset_path, task + '_llm.pkl')
        else:
            # pkl_file = os.path.join(dataset_path, task + '_full.pkl')
            pkl_file = os.path.join(dataset_path, task + '.pkl')
            # pkl_file = os.path.join(dataset_path, task + '_test_corner_cases.pkl')
            # pkl_file = os.path.join(dataset_path, task + '_test_format_heterogeneity.pkl')
            # pkl_file = os.path.join(dataset_path, task + '_test_missing_values.pkl')
            # pkl_file = os.path.join(dataset_path, task + '_test_random.pkl')

        data_dict[task] = load(pkl_file)
        task_num_class_dict[task] = len(data_dict[task]['label2idx'])

    batch_size = args.batch_size

    valid_datasets = []
    valid_dataloaders = []
    test_datasets = []
    test_dataloaders = []

    for task in args.tasks:

        print (task)

        if task == "CTA-TURL":

            test_cta_turl_dataset = CTATestDataset(data_dict[task], cta_table_dict, 'test', args)
            test_cta_turl_dataloader = DataLoader(test_cta_turl_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_test_fn)

            valid_cta_turl_dataset = CTADataset(data_dict[task], cta_table_dict, 'dev', args)
            valid_cta_turl_dataloader = DataLoader(valid_cta_turl_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            test_datasets.append(test_cta_turl_dataset)
            test_dataloaders.append(test_cta_turl_dataloader)
            valid_datasets.append(valid_cta_turl_dataset)
            valid_dataloaders.append(valid_cta_turl_dataloader)

        elif task == "CPA-TURL":

            test_cpa_turl_dataset = CPATestDataset(data_dict[task], cpa_table_dict, 'test', args)
            test_cpa_turl_dataloader = DataLoader(test_cpa_turl_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_test_fn)

            valid_cpa_turl_dataset = CPADataset(data_dict[task], cpa_table_dict, 'dev', args)
            valid_cpa_turl_dataloader = DataLoader(valid_cpa_turl_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            test_datasets.append(test_cpa_turl_dataset)
            test_dataloaders.append(test_cpa_turl_dataloader)
            valid_datasets.append(valid_cpa_turl_dataset)
            valid_dataloaders.append(valid_cpa_turl_dataloader)

        elif task == "CTA-GIT":

            test_cta_git_dataset = CTATestDataset(data_dict[task], cta_table_dict, 'test', args)
            test_cta_git_dataloader = DataLoader(test_cta_git_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_test_fn)

            valid_cta_git_dataset = CTADataset(data_dict[task], cta_table_dict, 'dev', args)
            valid_cta_git_dataloader = DataLoader(valid_cta_git_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            test_datasets.append(test_cta_git_dataset)
            test_dataloaders.append(test_cta_git_dataloader)
            valid_datasets.append(valid_cta_git_dataset)
            valid_dataloaders.append(valid_cta_git_dataloader)

        elif task == "CPA-GIT":

            test_cpa_git_dataset = CPATestDataset(data_dict[task], cpa_table_dict, 'test', args)
            test_cpa_git_dataloader = DataLoader(test_cpa_git_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_test_fn)

            valid_cpa_git_dataset = CPADataset(data_dict[task], cpa_table_dict, 'dev', args)
            valid_cpa_git_dataloader = DataLoader(valid_cpa_git_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            test_datasets.append(test_cpa_git_dataset)
            test_dataloaders.append(test_cpa_git_dataloader)
            valid_datasets.append(valid_cpa_git_dataset)
            valid_dataloaders.append(valid_cpa_git_dataloader)

        elif task == "CTA-SCH":

            test_cta_sch_dataset = CTATestDataset(data_dict[task], cta_table_dict, 'test', args)
            test_cta_sch_dataloader = DataLoader(test_cta_sch_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_test_fn)

            valid_cta_sch_dataset = CTADataset(data_dict[task], cta_table_dict, 'validation', args)
            valid_cta_sch_dataloader = DataLoader(valid_cta_sch_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            test_datasets.append(test_cta_sch_dataset)
            test_dataloaders.append(test_cta_sch_dataloader)
            valid_datasets.append(valid_cta_sch_dataset)
            valid_dataloaders.append(valid_cta_sch_dataloader)

        elif task == "CPA-SCH":

            test_cpa_sch_dataset = CPATestDataset(data_dict[task], cpa_table_dict, 'test', args)
            test_cpa_sch_dataloader = DataLoader(test_cpa_sch_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_test_fn)

            valid_cpa_sch_dataset = CPADataset(data_dict[task], cpa_table_dict, 'validation', args)
            valid_cpa_sch_dataloader = DataLoader(valid_cpa_sch_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            test_datasets.append(test_cpa_sch_dataset)
            test_dataloaders.append(test_cpa_sch_dataloader)
            valid_datasets.append(valid_cpa_sch_dataset)
            valid_dataloaders.append(valid_cpa_sch_dataloader)

        elif task == "CTA-DBP":

            test_cta_dbp_dataset = CTATestDataset(data_dict[task], cta_table_dict, 'test', args)
            test_cta_dbp_dataloader = DataLoader(test_cta_dbp_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_test_fn)

            valid_cta_dbp_dataset = CTADataset(data_dict[task], cta_table_dict, 'validation', args)
            valid_cta_dbp_dataloader = DataLoader(valid_cta_dbp_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            test_datasets.append(test_cta_dbp_dataset)
            test_dataloaders.append(test_cta_dbp_dataloader)
            valid_datasets.append(valid_cta_dbp_dataset)
            valid_dataloaders.append(valid_cta_dbp_dataloader)

        elif task == "CPA-DBP":

            test_cpa_dbp_dataset = CPATestDataset(data_dict[task], cpa_table_dict, 'test', args)
            test_cpa_dbp_dataloader = DataLoader(test_cpa_dbp_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_test_fn)

            valid_cpa_dbp_dataset = CPADataset(data_dict[task], cpa_table_dict, 'validation', args)
            valid_cpa_dbp_dataloader = DataLoader(valid_cpa_dbp_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

            test_datasets.append(test_cpa_dbp_dataset)
            test_dataloaders.append(test_cpa_dbp_dataloader)
            valid_datasets.append(valid_cpa_dbp_dataset)
            valid_dataloaders.append(valid_cpa_dbp_dataloader)

        else:
            raise ValueError("task name must be wrong.")

    for task, test_dataset in zip(args.tasks, test_datasets):
        print("{} Batch Total Num: {}".format(task, int(len(test_dataset) / (batch_size))))

    return valid_datasets, valid_dataloaders, test_datasets, test_dataloaders, data_dict, task_num_class_dict

if __name__ == "__main__":
    pass