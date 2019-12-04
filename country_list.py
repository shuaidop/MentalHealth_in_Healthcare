from lxml import html
import requests


def contry_cat(x):

	developing_clist=[]
	page = requests.get('https://isge2018.isgesociety.com/registration/list-of-developing-countries/')
	tree = html.fromstring(page.content)

	for num in range(1,51): 
	    c=tree.xpath('//*[@id="post-94"]/div/div/div[2]/div[1]/ul/li['+str(num)+']/text()')[0].replace(' ','')
	    developing_clist.append(c)
	for num in range(1,51): 
	    c=tree.xpath('//*[@id="post-94"]/div/div/div[2]/div[2]/ul/li['+str(num)+']/text()')[0].replace(' ','')
	    developing_clist.append(c)
	for num in range(1,46): 
	    c=tree.xpath('//*[@id="post-94"]/div/div/div[2]/div[3]/ul/li['+str(num)+']/text()')[0].replace(' ','')
	    developing_clist.append(c)

    if x in developing_clist:
        return 0
    else:
        return 1
