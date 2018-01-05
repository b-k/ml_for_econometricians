LDLIBS=-lm -lapophenia -lgsl
CFLAGS=-g --std=gnu99 -Wall -fopenmp -I. -I..

out/tab_checks/have_stats_tab: check_tabs out/tab_checks/have_remit_tab
	mkdir -p out/tab_checks
	. ./readin

out/tab_checks/have_remit_tab: check_tabs 
	mkdir -p out/tab_checks
	python -c 'import numpy; import pandas as pd; pd.read_csv("bilateralremittancematrix2016_Oct2017_updated.csv", index_col=0).stack().reset_index().to_csv("out/remit.csv", sep="|")' 
	#Pakistan/India have remittances of "0*"
	cat out/remit.csv | sed -e 's/\([0-9]\),\([0-9]\)/\1\2/' -e "s/Faeroe/Faroe/" -e "s/Korea, Dem. Rep./Korea, Dem. People’s Rep./"  -e "1 s/level_0/nameout/; s/level_1/namein/" -e '1s/0/remitted/' \
		-e '/Pakistan|India/d' -e '/India|Pakistan/d' \
	   	| apop_text_to_db -eo -d"|" - remit_tab remit.db
	touch out/tab_checks/have_remit_tabs

check_tabs:
	if sqlite3 remit.db "select * from  remit_tab limit 1" 2&>1 >/dev/null ; then touch out/tab_checks/have_remit_tab; fi
