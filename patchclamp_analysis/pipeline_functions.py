import os
import numpy as np
import pandas as pd
import h5py
import pyabf
from tqdm import tqdm


def analysis_iterator_h5(h5_data_loc, analyzer_configs):
    """
    Loop through recording groups in single_files, run analysis, save results to groups.

    Parameters:
    - h5_data_loc: path to HDF5 file
    - analyzer_configs: dict mapping protocol names to analyzer functions
                        e.g., {'IV': {'func': analyze_iv, 'arg1': val1}, ...}
    """

    with h5py.File(h5_data_loc, 'a') as f:
        abf_files = f['abf_files']

        for base_name in tqdm(list(abf_files.keys()), desc="Processing recordings"):
            grp = abf_files[base_name]

            # Get metadata from attributes
            filepath = grp.attrs['filepath']
            protocol = grp.attrs['protocol']

            print(f"\nProcessing: {base_name}")

            # Get appropriate analyzer and run
            if protocol in analyzer_configs:
                config = analyzer_configs[protocol].copy()
                analyzer_func = config.pop('func')
                abf = pyabf.ABF(filepath)
                results = analyzer_func(abf, **config)
            else:
                print(f"  No analyzer for protocol: {protocol}")
                results = {}

            # Save results to the group
            for key, val in results.items():
                # Delete if exists (overwrite)
                if key in grp:
                    del grp[key]

                if isinstance(val, dict):
                    # Nested dict -> dataset (structured array)
                    keys_arr = np.array(list(val.keys()))
                    vals_arr = np.array(list(val.values()))
                    dt = np.dtype([('key', keys_arr.dtype), ('value', vals_arr.dtype)])
                    data = np.array(list(zip(keys_arr, vals_arr)), dtype=dt)
                    grp.create_dataset(key, data=data)
                elif val is not None:
                    # Scalar or array -> dataset
                    grp.create_dataset(key, data=val)

    return None


def build_analysis_h5(dataset_info, overwrite=False):
    data_name = dataset_info['data_name']
    data_source = dataset_info['data_source']
    file_naming_scheme = dataset_info['file_naming_scheme']
    h5_filename = f'{data_name}_analysis_recs.h5'
    file_list = [os.path.join(dirpath, filename)
                 for dirpath, dirnames, filenames in os.walk(data_source)
                 for filename in filenames
                 if filename.endswith('.abf')]
    with h5py.File(h5_filename, 'a') as hf:
        if 'abf_files' in hf:
            single_files_group = hf['abf_files']
        else:
            single_files_group = hf.create_group('abf_files')
        for file_id, file_name in enumerate(file_list):
            file_metadata = dict()
            base_name = os.path.basename(file_name)
            file_metadata['recording_name'] = base_name
            parsed_name = parse_name(base_name, file_naming_scheme)
            file_metadata.update(parsed_name)
            # Check if this recording already exists in the group
            if base_name in single_files_group:
                if overwrite:
                    del single_files_group[base_name]
                else:
                    print(f'{base_name} already exists, skipping')
                    continue
            abf = pyabf.ABF(file_name)
            file_metadata['protocol'] = abf.protocol
            file_metadata['channelList'] = str(abf.channelList)
            file_metadata['abf_timestamp'] = str(abf.abfDateTime)
            rec_group = single_files_group.create_group(base_name)
            rec_group.attrs['file_id'] = file_id
            rec_group.attrs['filepath'] = str(file_name)
            for key, value in file_metadata.items():
                rec_group.attrs[key] = value
    print(f"Saved metadata for {len(file_list)} files to {h5_filename}")
    return h5_filename

def parse_name(base_name,scheme):
    parsed_name=dict()
    split_words = base_name.split('_')
    re_code = ['_'+split_words[i] for i in range(len(scheme))]
    re_code = ''.join(re_code)[1:]
    parsed_name['cell_id']= re_code
    for ci in range(len(scheme)):
        parsed_name[scheme[ci]] = split_words[ci]
    return parsed_name



def print_h5_tree(filepath, head=None, give_tree=False):
    def recurse(obj, prefix=""):
        lines = []
        keys = list(obj.keys())
        has_attrs = len(obj.attrs) > 0

        # Print attributes first if they exist
        if has_attrs:
            lines.append(f"{prefix}|-- [ATTRS]: {len(obj.attrs)} attributes")

        for i, key in enumerate(keys):
            item = obj[key]
            is_last = (i == len(keys) - 1)
            connector = "`-- " if is_last else "|-- "

            if isinstance(item, h5py.Group):
                item_has_attrs = len(item.attrs) > 0
                item_has_datasets = len(item.keys()) > 0

                if not item_has_datasets and not item_has_attrs:
                    lines.append(f"{prefix}{connector}[G] {key}/ (empty)")
                else:
                    lines.append(f"{prefix}{connector}[G] {key}/")
                    extension = "    " if is_last else "|   "
                    lines.extend(recurse(item, prefix + extension))
            else:
                lines.append(f"{prefix}{connector}[D] {key}: {item.shape} {item.dtype}")

        return lines

    with h5py.File(filepath, 'r') as f:
        tree_lines = [f.filename]
        tree_lines.extend(recurse(f))

    full_tree = '\n'.join(tree_lines)

    # Print only the specified number of lines
    if head is not None:
        lines_to_print = tree_lines[:head]
        print('\n'.join(lines_to_print))
        if len(tree_lines) > head:
            print(f"\n... ({len(tree_lines) - head} more lines)")
    else:
        print(full_tree)

    if give_tree:
        return full_tree
    else:
        return None

def restratify_h5_by_attribute(input_file, output_file, grouping_attr):
    """
    Reorganize HDF5 file grouping by a specific attribute

    Parameters:
    -----------
    input_file : str
        Path to input HDF5 file
    output_file : str
        Path to output HDF5 file
    grouping_attr : str
        Attribute name to group by (e.g., 'cell_id')

    Returns:
    --------
    str
        Path to output file
    """
    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:

        # Check if 'abf_files' group exists
        if 'abf_files' not in f_in:
            print("Error: 'abf_files' group not found in input file")
            return output_file

        abf_files = f_in['abf_files']

        # VERIFY Below
        # Collect groups by the specified attribute
        attr_map = {}
        for group_name in abf_files.keys():
            group = abf_files[group_name]
            if isinstance(group, h5py.Group) and grouping_attr in group.attrs:
                attr_value = group.attrs[grouping_attr]
                if attr_value not in attr_map:
                    attr_map[attr_value] = []
                attr_map[attr_value].append((group_name, group))
        # VERIFY ABOVE

        # Create new structure
        for attr_value, recordings in attr_map.items():
            # Use attribute value directly as group name
            new_group = f_out.create_group(str(attr_value))
            new_group.attrs[grouping_attr] = attr_value

            # Copy shared metadata from first recording
            first_rec = recordings[0][1]
            for key, value in first_rec.attrs.items():
                if key not in new_group.attrs:
                    new_group.attrs[key] = value

            # Merge all datasets from all recordings directly into the cell group
            for orig_name, orig_group in recordings:
                # Copy all datasets directly
                for dataset_name in orig_group.keys():
                    if isinstance(orig_group[dataset_name], h5py.Dataset):
                        target_name = dataset_name
                        counter = 1
                        while target_name in new_group:
                            target_name = f'{dataset_name}_{counter}'
                            counter += 1
                        f_in.copy(f'abf_files/{orig_name}/{dataset_name}', new_group, name=target_name)

        print(f"Restratified by '{grouping_attr}'")
        print(f"Created {len(attr_map)} groups")
        print(f"Total source recordings merged: {sum(len(recs) for recs in attr_map.values())}")
        print(f"Output: {output_file}")

    return output_file

def add_current_density_datasets(h5_filename, capacitance_key='Cmq_160.0', iv_datasets=['IV_K_130_140', 'IV_Na_0.2_10']):
    """
    Add current density datasets to HDF5 file groups.

    For each file group:
    1. Get capacitance value from dataset
    2. For each IV dataset, create a density version
    3. Divide current by capacitance (or fill with NaN if invalid)
    4. Save as new dataset with 'density_' prefix

    Parameters:
    - h5_filename: path to HDF5 file
    - capacitance_key: dataset name for capacitance (default 'Cmq_160.0')
    - iv_datasets: list of IV dataset names to convert

    Returns:
    - list of groups that failed
    """
    failed_groups = []
    total_groups = 0
    successful_groups = 0

    with h5py.File(h5_filename, 'a') as f:
        # Loop through all groups at the top level
        for group_name in f.keys():
            grp = f[group_name]

            # Skip if not a group
            if not isinstance(grp, h5py.Group):
                continue

            total_groups += 1

            # Get capacitance value from dataset (not attribute)
            capacitance_valid = False
            error_msg = None

            if capacitance_key not in grp:
                error_msg = f'{capacitance_key} not found'
            else:
                capacitance = grp[capacitance_key][()]
                if pd.isna(capacitance):
                    error_msg = f'{capacitance_key} is NaN'
                elif capacitance <= 0:
                    error_msg = f'{capacitance_key} is {capacitance} (invalid)'
                else:
                    capacitance_valid = True

            # Process each IV dataset
            group_success = False
            for iv_name in iv_datasets:
                if iv_name not in grp:
                    continue

                density_name = f'density_{iv_name}'
                if density_name in grp:
                    del grp[density_name]

                # Read original IV data (voltage, current) pairs
                iv_data = grp[iv_name][:]

                # Create new array
                density_data = np.zeros_like(iv_data)
                density_data['key'] = iv_data['key']  # Voltage stays same

                if capacitance_valid:
                    # Divide current by capacitance
                    density_data['value'] = iv_data['value'] / capacitance
                    group_success = True
                else:
                    # Fill with NaN
                    density_data['value'] = np.full(len(iv_data), np.nan)

                grp.create_dataset(density_name, data=density_data)

            if not group_success and error_msg:
                failed_groups.append((group_name, error_msg))
            else:
                successful_groups += 1

    print(f"Current density conversion: {successful_groups}/{total_groups} groups successful")

    return failed_groups


def get_datasets_from_h5(h5_file, dataset_names, attributes=None):
    """
    Extract datasets and attributes from HDF5 file as DataFrame

    Parameters:
    -----------
    h5_file : str
        Path to HDF5 file
    dataset_names : list or str
        Dataset name(s) to extract. Can be a single string or list of strings.
    attributes : list or str, optional
        Attribute(s) to include as columns. Can be a single string or list of strings.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: cell_id, attributes (if provided), and dataset_names
    """
    # Ensure inputs are lists
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    if isinstance(attributes, str):
        attributes = [attributes]
    elif attributes is None:
        attributes = []

    rows = []

    with h5py.File(h5_file, 'r') as hf:
        for group_name in hf.keys():
            group = hf[group_name]

            row = {'cell_id': group.attrs.get('cell_id', group_name)}

            # Extract requested attributes
            for attr in attributes:
                if attr in group.attrs:
                    row[attr] = group.attrs[attr]
                else:
                    row[attr] = np.nan

            # Extract requested datasets
            for dataset_name in dataset_names:
                if dataset_name in group:
                    value = group[dataset_name][()]
                    # Handle scalar arrays
                    if isinstance(value, np.ndarray) and value.size == 1:
                        value = value.item()
                    row[dataset_name] = value
                else:
                    row[dataset_name] = np.nan

            rows.append(row)
    df = pd.DataFrame(rows)
    return pd.DataFrame(rows)


def flatten_nested_column(df, group_col, nested_col):
    """
    Flatten a column containing list of [key, value] pairs.

    Parameters:
    - df: DataFrame with nested column
    - group_col: column to group by (e.g., 'cell_id', 'genotype')
    - nested_col: column with nested data like [[0, 5], [10, 12], ...]

    Returns:
    - dict of DataFrames, one per group
    """
    result_dfs = {}

    for group_name, group_df in df.groupby(group_col):
        # Collect all the nested data for this group
        rows_data = {}

        for idx, row in group_df.iterrows():
            cell_id = row.get('cell_id', idx)  # Or whatever identifier
            nested_data = row[nested_col]

            # Skip NaN/None entries
            if nested_data is None:
                continue

            # Check if it's a scalar NaN (not a list)
            if not isinstance(nested_data, (list, np.ndarray)):
                if pd.isna(nested_data):
                    continue

            # Skip empty lists
            if len(nested_data) == 0:
                continue

            # Convert [[key, val], [key, val]...] to dict
            for key, val in nested_data:
                if key not in rows_data:
                    rows_data[key] = {}
                rows_data[key][cell_id] = val

        # Convert to DataFrame
        result_dfs[group_name] = pd.DataFrame(rows_data).T.sort_index()

    return result_dfs


def write_dict_to_excel(result_dfs, filename):
    """
    Write dictionary of DataFrames to Excel with each DataFrame as a separate sheet.

    Parameters:
    - result_dfs: dict where keys are sheet names, values are DataFrames
    - filename: output Excel filename (e.g., 'results.xlsx')
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, df in result_dfs.items():
            df.to_excel(writer, sheet_name=str(sheet_name))

    print(f"Wrote {len(result_dfs)} sheets to {filename}")
    print(os.getcwd())
    return None
