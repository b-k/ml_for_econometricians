Common techniques across econometrics and ML
==========

This repository holds the slides and scripts for a talk I (BK) gave at the IMF as part of an ML symposium in November 2017. Here's the abstract for the talk:

    Characterizing ML methods as a quantum leap from traditional statistical methods sells
    textbooks and software, but there is a smooth transition between the various methods. This
    workshop begins with a discussion of the commonalities across all models and universal
    techniques available for validation. We then apply these techniques to models of
    international remittances across the spectrum from regressions to neural networks.

The examples use various measures to fit models to remittance data. I put these models
together for this talk, so the process for downloading the data and fitting the models
are in this repository as well. If you'd like to collaborate on formalizing and extending the analysis,
please contact me.

The slide carousel
------

The main directory holds the sequence of slides for the talk, many of which are linked
to some point in the demo script. A PDF version is included in this repository along
with the elements needed to build the PDF. Those elements are written for XeTeX (a
unicode-friendly version of LaTeX, which should install via your package manager). If
you are on a POSX-compatible system, run `make` in the home directory to rebuild.

The data and analysis
-------

If you run `make` in the `modelfits` directory, the scripts will run to download the
data and generate the data set for the demo. You'll need Pandas for Python, curl, unzip,
sqlite3, and Apophenia, which you may be able to `apt-get` into your system. If you
don't want to use Apophenia, replace the `apop_text_to_db` commands with your preferred
CSV-to-SQLite readin script.

The script will spend the bulk of its time in the data download and read-in of
those data sets. If that completed but the database building failed, comment out the
`download_and_read` line toward the bottom of the script to not repeat the downloads.

The Python 3 script, also in the `modelfits` directory, is named `go.py`. You can get the package
dependencies, primarily scipy and sklearn, from the imports at the head of the script.

Watch this space!
-------

I've been asked to give the talk at IMF again on 14 Feb 2018, and I'll be making revisions based on the feedback from the first talk (notably, more on causality), and to clean up the code posted here.
