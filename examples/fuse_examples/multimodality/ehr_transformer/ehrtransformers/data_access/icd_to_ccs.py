"""
Load ICD to CCS mapping from RWD database and save to json.

Requires db credentials stored in env vars `DB_USER` and `DB_PASSWORD`.
This needs to be run only once to create the json.

The table that we are using is created here:
https://github.ibm.com/MLHLS/rwd-processing/blob/45b3af5334da46240f27d125f5503d0ee79a08a9/prep/preprocessing/sqlTruven/createIcdCcsTables.sql

Note that because CCS is not defined for icd10, the icd10 keys are mapped first
to icd9 and then to CCS. Note also that the icd9 "E" codes overlap with icd10
codes. There are 35 like that. Here we utilize the icd9 version because the
majority of the data is icd9. Should that assumption change, this script would
need to be modified.
"""
from os import environ
import os
import pandas as pd
# from ibm_db import connect
# from ibm_db_dbi import Connection
import json

# props = {
#     "HOSTNAME": "lnx-vfn106.haifa.ibm.com",
#     "PORT": "50000",
#     "DATABASE": "Truven1",
#     "PROTOCOL": "TCPIP",
#     "UID": environ["DB_USER"],
#     "PWD": environ["DB_PASSWORD"],
# }
# connection_string = ";".join("%s=%s" % (k, v) for k, v in props.items())


def get_icd_ccs_dict(icd_ccs_map_df):
    icd_ccs_dict = icd_ccs_map_df.set_index("CODE")["CCS_CATEGORY"].to_dict()
    repeat_codes_dict = get_repeat_codes_dict(icd_ccs_map_df)
    icd_ccs_dict.update(repeat_codes_dict)
    return icd_ccs_dict


def get_repeat_codes_dict(icd_ccs_map_df, icd_to_use="icd9"):
    repeat_codes = icd_ccs_map_df[icd_ccs_map_df.duplicated("CODE")]
    repeat_codes = repeat_codes[repeat_codes.ICD_VERSION == icd_to_use]
    return repeat_codes.set_index("CODE")["CCS_CATEGORY"].to_dict()

def read_icd_ccs_dict(json_filepath=None):
    if json_filepath is None:
        log_path = os.path.abspath(__file__)
        code_path = os.path.dirname(os.path.dirname(os.path.dirname(log_path)))
        json_filepath = os.path.join(code_path, 'ehrtransformers/data_access/icd_to_ccs.json')

    with open(json_filepath) as f:
        tmp = json.load(f)
        return tmp

if __name__ == "__main__":

    a=1
    # connection = connect(connection_string, "", "")
    # conn = Connection(connection)
    # query = "SELECT * FROM RWD.v_icd_ccs_map icm"
    # icd_ccs_table = pd.read_sql(query, conn).applymap(lambda x: x.strip())
    # icd_to_ccs = get_icd_ccs_dict(icd_ccs_table)
    #
    # ofname = "icd_to_ccs.json"
    #
    # with open(ofname, "w", encoding="utf8") as f:
    #     json.dump(icd_to_ccs, f)
