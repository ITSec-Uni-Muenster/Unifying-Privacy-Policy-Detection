import logging

from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By

from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket

from bs4 import BeautifulSoup, SoupStrainer
import lxml
import cchardet as chardet
import pathlib
from selenium.webdriver.common.by import By
import requests
from urllib.parse import urljoin


class FindPrivacyPolicyCommand(BaseCommand):
    keywords_privacy = [ u"поверителност", u"политика за данни", u"политика лд", u"лични данни", u"бисквитки", u"условия", u"soukromí", u"používání dat", u"ochrana dat", u"osobních údajů", u"cookie", u"personlige oplysninger", u"datapolitik", u"privatliv", u"personoplysninger", u"regler om fortrolighed", u"personlige data", u"persondata", u"datenschutz", u"datenrichtlinie", u"privatsphäre", u"απορρήτου", u"απόρρητο", u"προσωπικά δεδομένα", u"εμπιστευτικότητας", u"ιδιωτικότητας", u"πολιτική δεδομένων", u"προσωπικών δεδομένων", u"privacy", u"data policy", u"data protection", u"privacidad", u"datos personales", u"política de datos", u"privaatsus", u"konfidentsiaalsus", u"isikuandmete", u"andmekaitse", u"küpsis", u"yksityisyy", u"tietokäytäntö", u"tietosuoja", u"henkilötie", u"eväste", u"confidentialite", u"confidentialité", u"vie privée", u"vie privee", u"données personnelles", u"donnees personnelles", u"utilisation des données", u"utilisation des donnees", u"rgpd", u"príobháideach", u"cosaint sonraí", u"cosanta sonraí", u"fianáin", u"fianán", u"privatnost", u"osobnih podataka", u"upotrebi podataka", u"zaštita podataka", u"obradi podataka", u"kolačić", u"adatvédel", u"adatkezel", u"személyes adatok védelme", u"riservatezza", u"privatezza", u"dati personali", u"privātum", u"sīkdat", u"privatum", u"konfidencialumas", u"asmens duomenų", u"duomenų sauga", u"slapuk", u"gegevensbescherming", u"gegevensbeleid", u"prywatnoś", u"dane osobowe", u"przetwarzanie danych", u"zasady przetwarzania danych", u"zasady dotyczące danych", u"ochrona danych", u"privacidade", u"dados pessoais", u"política de dados", u"rpgd", u"direitos do titular dos dados", u"confidențialitate", u"confidentialitate", u"protecția datelor", u"súkromi", u"využívania údajov", u"ochrana údajov", u"osobných údajov", u"zásady ochrany osobných", u"osobné údaje", u"gdpr", u"zasebnost", u"osebnih podatkov", u"piškotki", u"varstvo podatkov", u"sekretess", u"datapolicy", u"personuppgifter", u"integritet", u"kakor", u"informationskapslar", ]
    

    def __init__(self) -> None:
        self.logger = logging.getLogger("openwpm")
    
    def __repr__(self) -> str:
        return "FindPrivacyPolicyCommand"

    def url2filename(self, url):
        if url.lower().endswith('.php'):
            url = url.lower().replace('.php', '.html')
        return url.replace("http://","").replace("https://","").replace("'","").replace("/","_").replace(":","_").replace('www.', '')[0:150]


    def find_urls_with_list(self, c, site, a_tags):
        pp_urls = []
        for a_tag in a_tags:
            for p in self.keywords_privacy:
                if "href" in a_tag:
                    if p in a_tag["href"].lower(): 
                        if(self.modify_relative_urls(a_tag["href"], site) and self.modify_relative_urls(a_tag["href"], site) not in pp_urls):
                            pp_urls.append(self.modify_relative_urls(a_tag["href"], site))

        pp_urls_context = []
        for a_tag in a_tags:
            for p in self.keywords_privacy:
                try:
                    if p in a_tag.string.lower(): 
                        if(self.modify_relative_urls(a_tag["href"], site) and self.modify_relative_urls(a_tag["href"], site) not in pp_urls_context):
                            pp_urls_context.append(self.modify_relative_urls(a_tag["href"], site))
                except Exception as e:
                    pass
                
                try:
                    if len(pp_urls_context) == 0 and a_tag.previous_element.string:
                        if p in a_tag.previous_element.string.lower():
                            if(self.modify_relative_urls(a_tag["href"], site) and self.modify_relative_urls(a_tag["href"], site) not in pp_urls_context):
                                pp_urls_context.append(self.modify_relative_urls(a_tag["href"], site))
                except Exception as e:
                    self.logger.info(e)
                    pass
        merged = list( set(pp_urls + pp_urls_context))
        return merged


    def download_policies(self, pp_urls, manager_params, webdriver, site):
        pathlib.Path(str(manager_params.data_directory)+'/privacy_policies').mkdir(parents=True, exist_ok=True)
        pp_files = []

        for privacypolicy in pp_urls:
            download_file = self.download_privacy_policy(site, privacypolicy, str(manager_params.data_directory), webdriver)
            pp_files.append(download_file)

        self.logger.info("Privacy policies stored as:")
        self.logger.info(pp_files)

    def download_privacy_policy(self, site, url, path, driver):
        if url == "javascript":
            self.logger.info("Can't download JS links")
            return "javascript"
        else:
            driver.get(url)
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
                file.close()
            return path + "/privacy_policies/" + filename

    def modify_relative_urls(self, privacy_url, url):
        try:
            return urljoin(url, privacy_url)
        except Exception as e:
            print(url+' : '+privacy_url + ' URL not modified')
            return url+"/"+privacy_url


    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:

        src = webdriver.execute_script("return document.getElementsByTagName('html')[0].outerHTML")
        soup = BeautifulSoup(src, 'lxml')
        a_tags = soup.find_all('a')
        
        pp_urls = self.find_urls_with_list(self, webdriver.current_url, a_tags)
        
        self.logger.info(str(len(pp_urls)) + " potential privacy policy link(s) found on " + webdriver.current_url )

        self.logger.info(pp_urls)

        self.download_policies(pp_urls, manager_params, webdriver, webdriver.current_url)

        


