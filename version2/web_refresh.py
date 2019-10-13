from selenium import webdriver
import time
driver = webdriver.Chrome(executable_path="./chromedriver")
driver.get("file:///home/archisha/prototype-2.0/version2/output.html")
while True:
    driver.refresh()
    time.sleep(5)