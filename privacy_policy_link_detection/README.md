# Privacy Policy Link Detection

This script will automatically find links to (potential) privacy policies on websites. The script uses the most extensive approach to find as many candidates as possible. We use a broad list of search terms based on word list created for a study in 2019 you can find [here](https://github.com/RUB-SysSec/we-value-your-privacy/blob/master/privacy_wording.json) and search in link texts, urls and around the links. Once the files are downloaded you may use the rest of the toolchain to analyse the policies.

## Prerequisits

The privacy policy link detection provided here is implemented as a custom command for [openWPM](https://github.com/mozilla/OpenWPM).

To run our script you need to install additional pip requirements.
`conda activate openwpm && pip3 install lxml cchardet`

## Run Demo
Copy both files of this repo to the openWPM folder and run 
`python3 demo_privacy_policy_download.py`
Privacy policies identified on the specified websites will be stored in `datadir/privacypolices`.