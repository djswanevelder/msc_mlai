import os
import shutil

# The list of weight files provided by the user.
WEIGHT_FILES = [
    'ePZZH36doz3RPjmETdaTxo_67.pth', 'JSJkQKizxhwrFVpPtz3b3h_38.pth', 'F2WkFsEZ6GoPiLDQazAuq9_57.pth', 'Z9Wgk5q6PUhsN5bCCZbWzq_73.pth',
    'mBhkRxKsHnBYRPSr7Hbxsp_57.pth', 'WRQGZK2z2WgdChdDU5cBkd_44.pth', 'jaPLyma9tVfBQeCjsGr9MW_56.pth', '7bpJhWzwbdAFP8Wbz8vAi4_43.pth',
    'nKqPBdY6iMn9scYKxqZJct_13.pth', 'bgyUKNiPrvUTpJZAg4LXjq_18.pth', 'DprYVqsWJNKfyWfYpsrtrA_12.pth', '9RPMtNrv5wqTHcBpuqQxmJ_16.pth',
    '4qazTkXRYQa8xGbnDbcXeE_69.pth', 'U85ELXxJxEAry8EJCPu8wi_52.pth', 'eM2MkTS6x9uWUUWkBT4KbF_14.pth', 'hzcMnDq8QE47Ni96miQKUQ_10.pth',
    'gGSdGFYGZLNt4qh26LHgsw_12.pth', 'XX6QPKbXk8iv5v49B6hmjy_49.pth', 'Zhdme3mCqqZnYqms3VxLTi_44.pth', 'RGSKqWM4ZsRtxqptPjPhBu_18.pth',
    '5zUtmyD6sr64SCQDynuZpa_10.pth', 'ZZTCRjadANeZmhbAYA5D3e_15.pth', 'CjwsVETZ3MCLjcGUhganHA_41.pth', 'RLWSfuoAhxSACt2NEG6xCy_18.pth',
    'RrsFa56fpXy2RQE7ygioSe_42.pth', 'gRcwRQEdGqTWTeTJWiDmqd_21.pth', 'fhPDDsUyfJ6AYwtDwEWcSD_42.pth', 'kxNrByY452ceRV3sUiiffa_61.pth',
    'P3VegRLvhm5mCpzdTtRLxH_9.pth', 'TB47TAFZD8ZoWWfpRdvRwU_76.pth', 'MqBcBfKNR4vxu2Sras4JW7_48.pth', 'EeZ98m75VRPZrRHDxvejvd_43.pth',
    'WhhcYiZGkjb5Y5AN4Zo7RX_72.pth', 'Tp3W3AEKbt7oaqqzCKDp8Y_14.pth', 'fFUxenKM4yd6fopwn3qD5t_15.pth', 'ekDCZxna9Z5sVEHYpqiCeh_13.pth',
    'eMkQEiKCVCYf8XmNqVhPfb_29.pth', 'iB6XocRi3fUqdUmc7c3LdB_45.pth', 'dDyv43SjZ8d4WEoMTKiekr_14.pth', '4c6WfwmT4VNgdYvR5vzKD7_69.pth',
    '4Hwnhj4psanPs4Wdwxph9V_15.pth', 'gGSdGFYGZLNt4qh26LHgsw_48.pth', '532EHnro46vjXzHGgomhzD_12.pth', 'jy7iX9wbzT4m3oEbTDuJBq_9.pth',
    'eHnkzvkfjZuswc2t4v3s8w_16.pth', 'MppkioXhEa5xUt3BJKQymZ_16.pth', 'EPVKzUkxaqi4jaFAAoXrJt_16.pth', 'cpxK4Qcie8MRmBjFxdJ7um_18.pth',
    '5ZFPxQH8SunFTjNeNxAJJY_9.pth', 'TvcLonYMW8vsVurAeZqQJF_44.pth', '2jGPzZJrJNxU6ujivotyAi_27.pth', 'ePZZH36doz3RPjmETdaTxo_17.pth',
    'ekDCZxna9Z5sVEHYpqiCeh_80.pth', '5VPKMcijz5dd3cT5ojCG44_10.pth', 'JUoiGgEHtKqEJ48kXSW5EU_62.pth', '4qazTkXRYQa8xGbnDbcXeE_15.pth',
    '5vLswfJjve3JbjNebQCVex_9.pth', 'XMBs3TnEeo3Q9kzUjWaBcU_10.pth', 'eqgQz5SrLNgMiqPk2Db2ym_14.pth', 'Ba6gsyMXFaFEyBiAgLsaoo_15.pth',
    'mQarSuYK7vkRpJkJhscGVZ_9.pth', 'aPYWHvRkuUXh4qY8N4XEuE_75.pth', 'mWEy4mSg7FpQzUq92ZaPnj_18.pth', '9tMor4NNifAQ8xs2vvW666_60.pth', 'mpTyNaBs7GkBhBAbusY8rr_18.pth', 'JJ4jjJFkBNUBBNDWMGtggD_15.pth', 'oCxgYtPwQyUKojFq9GWqdA_18.pth', 'TB47TAFZD8ZoWWfpRdvRwU_18.pth',
    'Hfk6ebqjua8ve73r5UK7bX_59.pth', 'EEPneJg2zuDrrxGPwA5QoH_20.pth', 'WefoF4hVsuLP6vsKqHqtCn_49.pth', 'jJtsUMzrsAfu5NupBMutzw_13.pth',
    'fsprpwvqme6kgnxwkg6rtQ_59.pth', '4dWwLeJQRAj2DkBf39jANw_55.pth', 'VnaJ62XoxphbS9DfCsskTv_17.pth', 'bHCDjR8AP5EzNeGBaKqLRu_16.pth',
    'jWgjXd32hejMv5c3c7A9EY_50.pth', '6w4ztEb7JLALjgsaRHRcFg_11.pth', 'F2WkFsEZ6GoPiLDQazAuq9_11.pth', 'oEBE7furiAreUCPiLn3kJX_12.pth',
]

# Define the target directory where the files will be moved
DESTINATION_FOLDER = "data/weights/"

# Define the source directory where the files are currently located
INPUT_FOLDER = "data/weights/selected"

def move_files_to_folder(file_list, destination):
    """
    Creates a destination folder and moves all specified files into it.
    The source files are expected to be in the directory defined by INPUT_FOLDER.
    """
    print(f"Starting file movement process...")

    # 1. Create the destination folder if it doesn't exist
    try:
        os.makedirs(destination, exist_ok=True)
        print(f"Ensured directory '{destination}' exists.")
    except OSError as e:
        print(f"Error creating destination directory '{destination}': {e}")
        return

    # 2. Iterate through the files and move them
    files_moved = 0
    files_skipped = 0

    for filename in file_list:
        # Construct the full path using the new INPUT_FOLDER
        source_path = os.path.join(INPUT_FOLDER, filename)
        destination_path = os.path.join(destination, filename)

        try:
            # Check if the file exists before attempting to move
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
                print(f"Moved: '{source_path}' -> '{destination_path}'")
                files_moved += 1
            else:
                print(f"Skipped: '{source_path}' not found.")
                files_skipped += 1
        except Exception as e:
            print(f"Failed to move '{source_path}'. Error: {e}")
            files_skipped += 1

    print("\n--- Summary ---")
    print(f"Total files processed: {len(file_list)}")
    print(f"Successfully moved: {files_moved}")
    print(f"Files skipped (not found or error): {files_skipped}")

if __name__ == "__main__":
    move_files_to_folder(WEIGHT_FILES, DESTINATION_FOLDER)
