from scripts.all_imports import *


class Dataset_division(object):
    def __init__(self, args) -> None:
        self.args = args
    
    def divide_into_sections(self, val_dataset, test_dataset):
        """
        Divide a dataset into various sections: no_rule, one_rule, one_rule_contrast, one_rule_no_contrast etc.
        """

        val_dataset_no_rule = val_dataset.loc[val_dataset["rule_label"]==0]
        val_dataset_one_rule = val_dataset.loc[val_dataset["rule_label"]!=0]
        val_dataset_one_rule_contrast = val_dataset.loc[(val_dataset["rule_label"]!=0)&(val_dataset["contrast"]==1)]
        val_dataset_one_rule_no_contrast = val_dataset.loc[(val_dataset["rule_label"]!=0)&(val_dataset["contrast"]==0)]
        val_dataset_a_but_b = val_dataset.loc[val_dataset["rule_label"]==1]
        val_dataset_a_but_b_contrast = val_dataset.loc[(val_dataset["rule_label"]==1)&(val_dataset["contrast"]==1)]
        val_dataset_a_but_b_no_contrast = val_dataset.loc[(val_dataset["rule_label"]==1)&(val_dataset["contrast"]==0)]
        val_dataset_a_yet_b = val_dataset.loc[val_dataset["rule_label"]==2]
        val_dataset_a_yet_b_contrast = val_dataset.loc[(val_dataset["rule_label"]==2)&(val_dataset["contrast"]==1)]
        val_dataset_a_yet_b_no_contrast = val_dataset.loc[(val_dataset["rule_label"]==2)&(val_dataset["contrast"]==0)]
        val_dataset_a_though_b = val_dataset.loc[val_dataset["rule_label"]==3]
        val_dataset_a_though_b_contrast = val_dataset.loc[(val_dataset["rule_label"]==3)&(val_dataset["contrast"]==1)]
        val_dataset_a_though_b_no_contrast = val_dataset.loc[(val_dataset["rule_label"]==3)&(val_dataset["contrast"]==0)]
        val_dataset_a_while_b = val_dataset.loc[val_dataset["rule_label"]==4]
        val_dataset_a_while_b_contrast = val_dataset.loc[(val_dataset["rule_label"]==4)&(val_dataset["contrast"]==1)]
        val_dataset_a_while_b_no_contrast = val_dataset.loc[(val_dataset["rule_label"]==4)&(val_dataset["contrast"]==0)]

        test_dataset_no_rule = test_dataset.loc[test_dataset["rule_label"]==0]
        test_dataset_one_rule = test_dataset.loc[test_dataset["rule_label"]!=0]
        test_dataset_one_rule_contrast = test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]
        test_dataset_one_rule_no_contrast = test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==0)]
        test_dataset_a_but_b = test_dataset.loc[test_dataset["rule_label"]==1]
        test_dataset_a_but_b_contrast = test_dataset.loc[(test_dataset["rule_label"]==1)&(test_dataset["contrast"]==1)]
        test_dataset_a_but_b_no_contrast = test_dataset.loc[(test_dataset["rule_label"]==1)&(test_dataset["contrast"]==0)]
        test_dataset_a_yet_b = test_dataset.loc[test_dataset["rule_label"]==2]
        test_dataset_a_yet_b_contrast = test_dataset.loc[(test_dataset["rule_label"]==2)&(test_dataset["contrast"]==1)]
        test_dataset_a_yet_b_no_contrast = test_dataset.loc[(test_dataset["rule_label"]==2)&(test_dataset["contrast"]==0)]
        test_dataset_a_though_b = test_dataset.loc[test_dataset["rule_label"]==3]
        test_dataset_a_though_b_contrast = test_dataset.loc[(test_dataset["rule_label"]==3)&(test_dataset["contrast"]==1)]
        test_dataset_a_though_b_no_contrast = test_dataset.loc[(test_dataset["rule_label"]==3)&(test_dataset["contrast"]==0)]
        test_dataset_a_while_b = test_dataset.loc[test_dataset["rule_label"]==4]
        test_dataset_a_while_b_contrast = test_dataset.loc[(test_dataset["rule_label"]==4)&(test_dataset["contrast"]==1)]
        test_dataset_a_while_b_no_contrast = test_dataset.loc[(test_dataset["rule_label"]==4)&(test_dataset["contrast"]==0)]

        val_datasets = {"val_dataset":val_dataset, 
                        "val_dataset_no_rule":val_dataset_no_rule, 
                        "val_dataset_one_rule":val_dataset_one_rule, 
                        "val_dataset_one_rule_contrast":val_dataset_one_rule_contrast, 
                        "val_dataset_one_rule_no_contrast":val_dataset_one_rule_no_contrast, 
                        "val_dataset_a_but_b":val_dataset_a_but_b, 
                        "val_dataset_a_but_b_contrast":val_dataset_a_but_b_contrast, 
                        "val_dataset_a_but_b_no_contrast":val_dataset_a_but_b_no_contrast,
                        "val_dataset_a_yet_b":val_dataset_a_yet_b, 
                        "val_dataset_a_yet_b_contrast":val_dataset_a_yet_b_contrast, 
                        "val_dataset_a_yet_b_no_contrast":val_dataset_a_yet_b_no_contrast,
                        "val_dataset_a_though_b":val_dataset_a_though_b, 
                        "val_dataset_a_though_b_contrast":val_dataset_a_though_b_contrast, 
                        "val_dataset_a_though_b_no_contrast":val_dataset_a_though_b_no_contrast,
                        "val_dataset_a_while_b":val_dataset_a_while_b, 
                        "val_dataset_a_while_b_contrast":val_dataset_a_while_b_contrast, 
                        "val_dataset_a_while_b_no_contrast":val_dataset_a_while_b_no_contrast}
        
        test_datasets = {"test_dataset":test_dataset, 
                        "test_dataset_no_rule":test_dataset_no_rule, 
                        "test_dataset_one_rule":test_dataset_one_rule, 
                        "test_dataset_one_rule_contrast":test_dataset_one_rule_contrast, 
                        "test_dataset_one_rule_no_contrast":test_dataset_one_rule_no_contrast, 
                        "test_dataset_a_but_b":test_dataset_a_but_b, 
                        "test_dataset_a_but_b_contrast":test_dataset_a_but_b_contrast, 
                        "test_dataset_a_but_b_no_contrast":test_dataset_a_but_b_no_contrast,
                        "test_dataset_a_yet_b":test_dataset_a_yet_b, 
                        "test_dataset_a_yet_b_contrast":test_dataset_a_yet_b_contrast, 
                        "test_dataset_a_yet_b_no_contrast":test_dataset_a_yet_b_no_contrast,
                        "test_dataset_a_though_b":test_dataset_a_though_b, 
                        "test_dataset_a_though_b_contrast":test_dataset_a_though_b_contrast, 
                        "test_dataset_a_though_b_no_contrast":test_dataset_a_though_b_no_contrast,
                        "test_dataset_a_while_b":test_dataset_a_while_b, 
                        "test_dataset_a_while_b_contrast":test_dataset_a_while_b_contrast, 
                        "test_dataset_a_while_b_no_contrast":test_dataset_a_while_b_no_contrast}

        return val_datasets, test_datasets


    def train_val_test_split(self, dataset: dict, divide_into_rule_sections=False) -> dict:

        # convert into dataframe
        dataset = pd.DataFrame(dataset)

        # split the dataset into train, val and test
        train_idx, test_idx = train_test_split(list(range(dataset.shape[0])), test_size=0.2, random_state=self.args.seed_value)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=self.args.seed_value)
        train_dataset = dataset.iloc[train_idx].reset_index(drop=True)
        val_dataset = dataset.iloc[val_idx].reset_index(drop=True)
        test_dataset = dataset.iloc[test_idx].reset_index(drop=True)

        # divide val and test sets into rule sections
        if divide_into_rule_sections == True:
            val_datasets, test_datasets = self.divide_into_sections(val_dataset, test_dataset)

        # convert back into dict
        train_dataset = train_dataset.to_dict('list')
        for key, value in val_datasets.items():
            val_datasets[key] = val_datasets[key].to_dict('list')
        for key, value in test_datasets.items():
            test_datasets[key] = test_datasets[key].to_dict('list')

        return train_dataset, val_datasets, test_datasets