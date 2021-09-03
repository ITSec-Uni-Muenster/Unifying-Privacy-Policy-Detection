## Environment Setup


We use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) on Ubuntu 20.04 in order to create the Python environment. Please ensure that you have Java (e.g. openjdk-8-jre) installed. To setup the environment, please download `installation.sh` and run it like:

```
chmod +x installation.sh
./installation.sh
```

The tool expects a path like `data` followed by a folder named indicating the date of the crawl, which contains the rawl HTML/XML files of the respecting crawl by the downloader module. The tool reads these files and saves the data and according metadata in a [TinyDB](https://tinydb.readthedocs.io/en/latest/) database. If a different data storage format is required, the code can be adapted easily due to its modular design. If you have any questions about how to work with the code or regarding replication of our results, feel free to contact us.
