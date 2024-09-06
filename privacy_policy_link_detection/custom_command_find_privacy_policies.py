import logging
import traceback
import csv
import re
import os
import sqlite3
import sys
import ndjson
import multiprocessing as mp
import time
from time import sleep, localtime, strftime
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket
from bs4 import BeautifulSoup, SoupStrainer
from tranco import Tranco
import lxml
import cchardet as chardet
import pathlib
import requests
from datetime import datetime
from urllib.parse import urljoin



class FindPrivacyPolicyCommand(BaseCommand):
    # Keywords were expanded from "Comparing Large-Scale Privacy and Security Notifications" by Utz et al.
    keywords_privacy = [u"adatkezel", u"adatvédel", u"agb", u"andmekaitse", u"asmens duomenų", u"bedingungen", u"bảo mật", u"c.g.", u"cg", u"cgu", u"cgv", u"condicion", u"condiciones", u"conditii", u"conditions", u"condizioni", u"condições", u"confidentialitate", u"confidentialite", u"confidentialité", u"confidențialitate", u"cookie", u"cosaint sonraí", u"cosanta sonraí", u"dados pessoais", u"dane osobowe", u"data policy", u"data protection", u"datapolicy", u"datapolitik", u"datenrichtlinie", u"datenschutz", u"dati personali", u"datos personales", u"direitos do titular dos dados", u"disclaimer", u"donnees personnelles", u"données personnelles", u"duomenų sauga", u"eväste", u"feltételek", u"fianáin", u"fianán", u"galetes", u"gdpr", u"gegevensbeleid", u"gegevensbescherming", u"gizlilik", u"henkilötie", u"hinweis", u"informationskapslar", u"integritet", u"isikuandmete", u"jogi nyilatkozat", u"jogi tudnivalók", u"juriidili", u"kakor", u"ketentuan", u"kişisel verilerin", u"kolačić", u"konfidencialiteti", u"konfidencialumas", u"konfidentsiaalsus", u"koşulları", u"kvkk", u"käyttöehdot", u"küpsis", u"legal", u"légal", u"mbrojtja e të dhënave", u"nan", u"naudojimo taisyklės", u"naudotojo sutartis", u"noteikumi", u"obradi podataka", u"ochrana dat", u"ochrana údajov", u"ochrona danych", u"offenlegung", u"osebnih podatkov", u"osobnih podataka", u"osobné údaje", u"osobních údajů", u"osobných údajov", u"pedoman", u"persondata", u"personlige data", u"personlige oplysninger", u"personoplysninger", u"personuppgifter", u"personvern", u"persónuvernd", u"piškotki", u"podmienky", u"podmínky", u"pogoji", u"politica de utilizare", u"politika e privatësisë", u"politika e të dhënave", u"política de dados", u"política de datos", u"používání dat", u"pravidlá", u"pravila", u"pravno", u"privaatsus", u"privacidad", u"privacidade", u"privacitat", u"privacy", u"privasi", u"privatezza", u"privatliv", u"privatnost", u"privatsphäre", u"privatum", u"privātum", u"protecció de dades", u"protecția datelor", u"prywatnoś", u"przetwarzanie danych", u"príobháideach", u"quy chế", u"quy định", u"regler om fortrolighed", u"regulamin", u"rekisteriseloste", u"retningslinjer for data", u"rgpd", u"rgpd", u"riservatezza", u"rpgd", u"rules", u"sekretess", u"slapuk", u"sopimusehdot", u"soukromí", u"sutikimas", u"syarat", u"személyes adatok védelme", u"súkromi", u"sīkdat", u"teisinė", u"temeni", u"termene", u"termeni", u"termini", u"termos", u"terms", u"tiesību", u"tietokäytäntö", u"tietosuoja", u"tingimused", u"téarmaí", u"upotrebi podataka", u"utilisation des donnees", u"utilisation des données", u"uvjeti", u"varstvo podatkov", u"veri ilkesi", u"veri politikası", u"vie privee", u"vie privée", u"vilkår", u"villkor", u"voorwaarden", u"využívania údajov", u"warunki", u"yasal", u"yksityisyy", u"zasady dotyczące danych", u"zasady przetwarzania danych", u"zasebnost", u"zaštita podataka", u"zásady ochrany osobných", u"çerez", u"điều khoản", u"şartları", u"απορρήτου", u"απόρρητο", u"εμπιστευτικότητας", u"ιδιωτικότητας", u"πολιτική δεδομένων", u"προσωπικά δεδομένα", u"προσωπικών δεδομένων", u"όροι", u"бисквитки", u"конфиде", u"конфиденциальность", u"конфіденційність", u"лични данни", u"персональных данных", u"поверителност", u"политика за данни", u"политика использования", u"политика лд", u"политика о подацима", u"пользовательское соглашение", u"правила", u"приватност", u"споразумение", u"условия", u"הסכם שימוש", u"מדיניות נתונים", u"פרטיות", u"תנאי שימוש", u"תקנון", u"الخصوصية", u"حریم خصوصی", u"سياسة البيانات", u"شرایط و قوانین", u"قوانین و مقررات", u"ข้อกำหนดการใช้งาน", u"ข้อกำหนดของการบริการ" u"ข้อตกลงและเงื่อนไข", u"ความเป็นส่วนตัว", u"นโยบายความเป็นส่วนตัว", u"นโยบายคุกกี้", u"ประกาศนโยบายความเป็นส่วนตัว", u"เงื่อนไขและข้อกำหนด", u"ご利用上の注意", u"クッキー", u"プライバシー", u"個人情報", u"数据使用", u"數據使用", u"私隱", u"規約", u"隐私权", u"개인정보", u"이용약관 ", u"프라이버시"]
   
    def __init__(self, domain) -> None:
        self.logger = logging.getLogger("openwpm")
        self.domain = domain

    def __repr__(self) -> str:
        return "FindPrivacyPolicyCommand"


    def url2filename(self, url):
        if url.lower().endswith('.php'):
            url = url.lower().replace('.php', '.html')
        return url.replace("http://", "").replace("https://", "").replace("'", "").replace("/", "_").replace(":", "_").replace('www.', '')[0:150]


    def modify_relative_urls(self, privacy_url, url):
        try:
            return urljoin(url, privacy_url)
        except Exception as e:
            self.logger.error(url + ' : ' + privacy_url + ' URL not modified')
            return url + "/" + privacy_url


    def find_pp_urls_with_list(self, c, site, a_tags):
        pp_urls = []
        pp_urls_context = []

        self.logger.info("There are %d a_tags on %s", len(a_tags), site)

        for a_tag in a_tags:
            for p in self.keywords_privacy:
                try:
                    if "href" in a_tag.attrs and a_tag["href"] is not None:
                        if p in a_tag["href"].lower():
                            if self.modify_relative_urls(a_tag["href"], site) not in pp_urls:
                                pp_urls.append(self.modify_relative_urls(a_tag["href"], site))
                except Exception as e:
                    self.logger.error(e)
                    self.logger.error(traceback.print_exc())

                # context
                try:
                    if "href" in a_tag.attrs and a_tag["href"] is not None:
                        if a_tag.string:
                            if p in a_tag.string.lower(): 
                                if self.modify_relative_urls(a_tag["href"], site) not in pp_urls_context:
                                    pp_urls_context.append(self.modify_relative_urls(a_tag["href"], site))
                except Exception as e:
                    self.logger.error(e)
                    self.logger.error(traceback.print_exc())
                
                try:
                    if "href" in a_tag.attrs and a_tag["href"] is not None:
                        if len(pp_urls_context) == 0 and a_tag.previous_element.string:
                            if p in a_tag.previous_element.string.lower():
                                if self.modify_relative_urls(a_tag["href"], site) not in pp_urls_context:
                                    pp_urls_context.append(self.modify_relative_urls(a_tag["href"], site))
                except Exception as e:
                    self.logger.error(e)
                    self.logger.error(traceback.print_exc())
                    
        merged = list(set(pp_urls + pp_urls_context))
        return merged


    def find_california_urls(self, c, site, a_tags):
        california_privacy_links = []

        do_not_sell_link = r"^(ccpa|cpra|ca(lifornia)?)?\s?[-:]?\s?((do not)|(don't))\ssell\s?(my)?\s?(personal)?\s?(info(rmation)?|data|PI)?\s?(\(ca(lifornia)?\)|\(ccpa\))?$"
        privacy_notice_rights = r"^(?:(your)?\s?(ccpa|ca(lifornia)?)\s?privacy\s(notice|rights)|(privacy\snotice\sfor\sca\sresidents))$"
        californian_residents = r"^(ca(lifornia(n)?)?)\sresident(s)?$"

        for a_tag in a_tags:
            try:
                if "href" in a_tag.attrs and a_tag["href"] is not None:
                    a_tag_lowercase = a_tag["href"].lower()
                    list_of_matches = []
                    for regex in [do_not_sell_link, privacy_notice_rights, californian_residents]:
                        matches = re.finditer(regex, a_tag_lowercase, re.IGNORECASE | re.DOTALL | re.UNICODE | re.MULTILINE)
                        for match in matches:
                            temp = match.group()
                            temp = re.sub(" +", " ", temp).strip()
                            list_of_matches.append(temp)
                    if len(set(list_of_matches)) > 0:
                        if self.modify_relative_urls(a_tag["href"], site) not in california_privacy_links:
                                california_privacy_links.append(self.modify_relative_urls(a_tag["href"], site))

                    if (((('california' in a_tag_lowercase) and ('privacy' in a_tag_lowercase))
                        or (('california' in a_tag_lowercase) and ('right' in a_tag_lowercase)))
                        or ((('kaliforni' in a_tag_lowercase) and ('datenschutz' in a_tag_lowercase))
                        or (('kaliforni' in a_tag_lowercase) and ('rechte' in a_tag_lowercase)))):
                        if self.modify_relative_urls(a_tag["href"], site) not in california_privacy_links:
                                california_privacy_links.append(self.modify_relative_urls(a_tag["href"], site))
            except Exception as e:
                self.logger.error(e)
                self.logger.error(traceback.print_exc())

            # context
            try:
                if "href" in a_tag.attrs and a_tag["href"] is not None:
                    if a_tag.string:
                        a_tag_string_lowercase = a_tag.string.lower()
                        list_of_matches = []
                        for regex in [do_not_sell_link, privacy_notice_rights, californian_residents]:
                            matches = re.finditer(regex, a_tag_string_lowercase, re.IGNORECASE | re.DOTALL | re.UNICODE | re.MULTILINE)
                            for match in matches:
                                temp = match.group()
                                temp = re.sub(" +", " ", temp).strip()
                                list_of_matches.append(temp)
                        if len(set(list_of_matches)) > 0:
                            if self.modify_relative_urls(a_tag["href"], site) not in california_privacy_links:
                                    california_privacy_links.append(self.modify_relative_urls(a_tag["href"], site))
    
                        if(((('california' in a_tag_string_lowercase) and ('privacy' in a_tag_string_lowercase))
                        or (('california' in a_tag_string_lowercase) and ('right' in a_tag_string_lowercase)))
                        or ((('kaliforni' in a_tag_string_lowercase) and ('datenschutz' in a_tag_string_lowercase))
                        or (('kaliforni' in a_tag_string_lowercase) and ('rechte' in a_tag_string_lowercase)))):
                            if self.modify_relative_urls(a_tag["href"], site) not in california_privacy_links:
                                    california_privacy_links.append(self.modify_relative_urls(a_tag["href"], site))
            except Exception as e:
                self.logger.error(e)
                self.logger.error(traceback.print_exc())

            try:
                if "href" in a_tag.attrs and a_tag["href"] is not None:
                    if a_tag.previous_element.string:
                        a_tag_prev_element = a_tag.previous_element.string
                        list_of_matches = []
                        for regex in [do_not_sell_link, privacy_notice_rights, californian_residents]:
                            matches = re.finditer(regex, a_tag_prev_element, re.IGNORECASE | re.DOTALL | re.UNICODE | re.MULTILINE)
                            for match in matches:
                                temp = match.group()
                                temp = re.sub(" +", " ", temp).strip()
                                list_of_matches.append(temp)
                        if len(set(list_of_matches)) > 0:
                            if self.modify_relative_urls(a_tag["href"], site) not in california_privacy_links:
                                california_privacy_links.append(self.modify_relative_urls(a_tag["href"], site))
            except Exception as e:
                self.logger.error(e)
                self.logger.error(traceback.print_exc())

        return california_privacy_links


    def download_policies(self, pp_urls, manager_params, webdriver, site):
        pathlib.Path(str(manager_params.data_directory) + "/privacy_policies").mkdir(parents=True, exist_ok=True)
        pp_files = []

        for privacypolicy in pp_urls:
            download_file = self.download_privacy_policy(site, privacypolicy, str(manager_params.data_directory), webdriver)
            pp_files.append(download_file)

        self.logger.info("Privacy policies stored as:")
        self.logger.info(pp_files)
        return pp_files


    def download_privacy_policy(self, site, url, path, driver):
        if url == "javascript":
            self.logger.info("Can't download JS links")
            return "javascript"
        else:
            driver.get(url)
            driver.implicitly_wait(100)
            filename = self.url2filename(site + "_" + url)
            # Download privacy policies if they are files, otherwise store the content
            if url.endswith("pdf") or url.endswith("doc") or url.endswith("docx") or url.endswith("odt"):
                r = requests.get(url, stream=True)
                with open(path + "/privacy_policies/" + filename, 'wb') as file:
                    file.write(r.content)
                file.close()
            else:
                src = driver.execute_script("return document.getElementsByTagName('html')[0].outerHTML")
                allhtml = str(src)
                with open(path + "/privacy_policies/" + filename, 'w+') as file:
                    file.write(allhtml)
            return path + "/privacy_policies/" + filename


    def download_california_pages(self, cali_urls, manager_params, webdriver, site):
        pathlib.Path(str(manager_params.data_directory) + "/california_privacy_pages").mkdir(parents=True, exist_ok=True)
        cali_files = []

        for url in cali_urls:
            download_file = self.download_california_privacy_pages(site, url, str(manager_params.data_directory), webdriver)
            cali_files.append(download_file)

        self.logger.info("California pages stored as:")
        self.logger.info(cali_files)
        return cali_files


    def download_california_privacy_pages(self, site, url, path, driver):
        if url == "javascript":
            self.logger.info("Can't download JS links")
            return "javascript"
        else:
            driver.get(url)
            driver.implicitly_wait(100)
            filename = self.url2filename(site + "_" + url)
            # Download california policies if they are files, otherwise store the content
            if url.endswith("pdf") or url.endswith("doc") or url.endswith("docx") or url.endswith("odt"):
                r = requests.get(url, stream=True)
                with open(path + "/california_privacy_pages/" + filename, 'wb') as file:
                    file.write(r.content)
                file.close()
            else:
                src = driver.execute_script("return document.getElementsByTagName('html')[0].outerHTML")
                allhtml = str(src)
                with open(path + "/california_privacy_pages/" + filename, 'w+') as file:
                    file.write(allhtml)
                file.close()
            return path + "/california_privacy_pages/" + filename


    def save_landing_page(self, site, url, src, manager_params):
        pathlib.Path(str(manager_params.data_directory) + "/landing_pages").mkdir(parents=True, exist_ok=True)
        filename = self.url2filename(site + "_" + url)
        allhtml = str(src)
        with open(str(manager_params.data_directory) + "/landing_pages/" + filename, 'w+') as file:
            file.write(allhtml)
        lp_file = str(manager_params.data_directory) + "/landing_pages/" + filename
        self.logger.info("Landing page stored as:")
        self.logger.info(lp_file)
        return lp_file


    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:

        landing_url = webdriver.current_url
        src = webdriver.execute_script("return document.getElementsByTagName('html')[0].outerHTML")
        soup = BeautifulSoup(src, 'lxml')
        a_tags = soup.find_all('a')
        lp_file = self.save_landing_page(self.domain, landing_url, src, manager_params)

        self.logger.info("Starting to find privacy policy URLs")
        pp_urls = self.find_pp_urls_with_list(self, landing_url, a_tags)
        self.logger.info("Finished finding privacy policy URLs")
        self.logger.info(str(len(pp_urls)) + " potential privacy policy link(s) found on " + landing_url)
        self.logger.info(pp_urls)
        pp_files = self.download_policies(pp_urls, manager_params, webdriver, landing_url)

        self.logger.info("Starting to find California policy URLs")
        cali_urls = self.find_california_urls(self, landing_url, a_tags)
        self.logger.info("Finished finding California policy URLs")
        self.logger.info(str(len(cali_urls)) + " potential California link(s) found on " + landing_url)
        self.logger.info(cali_urls)
        cali_files = self.download_california_pages(cali_urls, manager_params, webdriver, landing_url)


        current_url_json = {'domain': self.domain,
           'crawl_id': browser_params.browser_id,
            'landing_url': landing_url,
            'landing_page': lp_file,
            'privacy_policy_url': pp_urls,
            'privacy_policy_file': pp_files,
            'california_url': cali_urls,
            'california_file': cali_files
            }
        try:
            with open(str(manager_params.data_directory) + "/crawl-data.ndjson", encoding="utf-8", mode="a") as f:
                writer = ndjson.writer(f, ensure_ascii=False)
                writer.writerow(current_url_json)
            self.logger.info("Saved all policies of " + landing_url)
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.print_exc())
