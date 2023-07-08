import os
import sys
from pathlib import Path
from datetime import datetime
import traceback
import time
import multiprocessing as mp
from tranco import Tranco
import ndjson 
import json

from custom_command_find_privacy_policies import FindPrivacyPolicyCommand
from openwpm.command_sequence import CommandSequence
from openwpm.commands.browser_commands import GetCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.task_manager import TaskManager

today = str(datetime.now().date())
today = "2023-06-13"

print("Default encoding:", sys.getdefaultencoding(), flush=True)

# The list of sites that we wish to crawl
NUM_BROWSERS = 10
tranco_date = "2022-12-23" # do not touch this during each series of collection
start_tranco = 0 # start if continued crawl.
amount_tranco = 100000 #do not touch this either.

path_res = "./datadir_" + today + "/"


def collect_tranco(input_date, start_tranco, amount_urls):
    donedomains = []
    tranco_urls = []
    print("Reading" + path_res + "crawl-data.ndjson", flush=True)
    if os.path.isfile(path_res + "crawl-data.ndjson"):
        with open(path_res + "crawl-data.ndjson", encoding="utf-8", mode="r") as f:
            # pp = ""
            # prev_pp = ""
            for i, line in enumerate(f, start=1):
                # catch eventual errros without the whole reading process breaking
                pp = None
                while pp is None:
                    try:
                        # prev_pp = pp
                        pp = json.loads(line.strip())
                    except json.decoder.JSONDecodeError as e: 
                        # traceback.print_exc()
                        print("Error in line", i)
                        # print(prev_pp["crawl_id"], flush=True)
                        # print(prev_pp["domain"], flush=True)
                        print(line[e.pos-5:e.pos+5])
                        pp = json.loads(line[e.pos-2:].strip())
                donedomains.append(pp["domain"])
            donedomains = list(set(donedomains))

    print(str(len(donedomains)) + " domains already checked", flush=True)
    tr = Tranco(cache=True, cache_dir=os.path.expanduser('~/tranco/'))
    # tranco_urls = mp.Manager().list()
    # temp = mp.Manager().list()
    if input_date == "default":
        temp = tr.list().top(start=start_tranco, num=amount_urls)
    else:
        temp = tr.list(date=input_date).top(start=start_tranco, num=amount_urls)
    skipped = 0
    for i in temp:
        if 'http://' + i not in donedomains:
            tranco_urls.append('http://' + i)
        else:
            skipped = skipped + 1
    print("Skipped " + str(skipped) + " URLs", flush=True)
    return tranco_urls

print("Start time: ", str(datetime.now()), flush=True)
t0 = time.time()
domains = collect_tranco(tranco_date, start_tranco, amount_tranco)
print('Number of domains to crawl: ', len(domains), flush=True)

# Loads the default ManagerParams
# and NUM_BROWSERS copies of the default BrowserParams

manager_params = ManagerParams(num_browsers=NUM_BROWSERS)
browser_params = [BrowserParams(display_mode="xvfb") for _ in range(NUM_BROWSERS)]

# Update browser configuration (use this for per-browser settings)
for browser_param in browser_params:
    # Record HTTP Requests and Responses
    browser_param.http_instrument = False
    # Record cookie changes
    browser_param.cookie_instrument = True
    # Record Navigations
    browser_param.navigation_instrument = True
    # Record JS Web API calls
    # see: https://github.com/openwpm/OpenWPM/issues/947
    # set to False if hangs
    browser_param.js_instrument = False 
    # Record the callstack of all WebRequests made
    browser_param.callstack_instrument = False
    # Record DNS resolution
    browser_param.dns_instrument = False
    # Bot Mitigations
    browser_param.bot_mitigation = True
    # Language Detection
    browser_param.languages = "en-us,en,de"
    browser_param.tp_cookies = "always"

# Update TaskManager configuration (use this for crawl-wide settings)
manager_params.data_directory = Path(path_res)
manager_params.log_path = Path(path_res + "openwpm.log")

# memory_watchdog and process_watchdog are useful for large scale cloud crawls.
# Please refer to docs/Configuration.md#platform-configuration-options for more information
manager_params.memory_watchdog = True
manager_params.process_watchdog = True
manager_params.failure_limit = 32768

# Commands time out by default after 60 seconds
with TaskManager(
    manager_params,
    browser_params,
    SQLiteStorageProvider(Path(path_res + "crawl-data.sqlite")),
    None,
) as manager:
    # Visits the sites
    for index, domain in enumerate(domains):

        def callback(success: bool, val: str = domain) -> None:
            print(
                f"CommandSequence for {val} ran {'successfully' if success else 'unsuccessfully'}", flush=True
            )

        # Parallelize sites over all number of browsers set above.
        command_sequence = CommandSequence(
            domain,
            reset=True,
            site_rank=index,
            callback=callback,
        )

        # Start by visiting the page
        command_sequence.append_command(GetCommand(url=domain, sleep=0), timeout=60)
        # Have a look at custom_command.py to see how to implement your own command
        command_sequence.append_command(FindPrivacyPolicyCommand(domain), timeout=60)
        # Run commands across all browsers (simple parallelization)
        manager.execute_command_sequence(command_sequence)

print("End time: ", str(datetime.now()), flush=True)
print(f'Run time: {(time.time() - t0):.3f}', flush=True)

