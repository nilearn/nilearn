.. _megatrawls_maps:

MegaTrawls Network Matrices HCP
===============================

Access
------
See :func:`nilearn.datasets.fetch_megatrawls_netmats`.

Notes
-----
Contains network matrices data of two types, full correlation and partial
correlation which were estimated using each subject specific timeseries
signals extracted from group of :term:`ICA` nodes or parcellations. In total,
461 functional connectivity datasets were used to obtain these matrices
and is part of HCP Megatrawls release.

The number of nodes available for download are 25, 50, 100, 200, 300
with combination of two variants of timeseries extraction methods,
multiple spatial regression (ts2) and eigen regression (ts3).

These matrices can be used to predict the relationships between subjects
functional connectivity datasets and their behavioral measures. Both can be
downloaded from HCP connectome website under conditions. See disclaimer below.

More information available in :footcite:t:`Smith2015b`,
:footcite:t:`Smith2015a`, :footcite:t:`Filippini2009`,
:footcite:t:`Smith2014`, and :footcite:t:`Reilly2009`.

Content
-------
    :'dimensions': contains given input in dimensions used in fetching data.
    :'timeseries': contains given specific timeseries method used in fetching data.
    :'matrices': contains given specific type of matrices name.
    :'correlation_matrices': contains correlation network matrices data.

References
----------

.. footbibliography::

For more technical details about predicting the measures, refer to:
Stephen Smith et al, HCP beta-release of the Functional Connectivity MegaTrawl.
April 2015 "HCP500-MegaTrawl" release.
https://db.humanconnectome.org/megatrawl/

Disclaimer
----------
IMPORTANT: This is open access data. You must agree to Terms and conditions
of using this data before using it, available at:
http://humanconnectome.org/data/data-use-terms/open-access.html

Open Access Data (all imaging data and most of the behavioral data)
is available to those who register an account at ConnectomeDB and agree to
the Open Access Data Use Terms. This includes agreement to comply with
institutional rules and regulations. This means you may need the approval
of your IRB or Ethics Committee to use the data. The released HCP data are
not considered de-identified, since certain combinations of HCP Restricted
Data (available through a separate process) might allow identification of
individuals. Different national, state and local laws may apply and be
interpreted differently, so it is important that you consult with your IRB
or Ethics Committee before beginning your research. If needed and upon
request, the HCP will provide a certificate stating that you have accepted the
HCP Open Access Data Use Terms. Please note that everyone who works with HCP
open access data must review and agree to these terms, including those who are
accessing shared copies of this data. If you are sharing HCP Open Access data,
please advice your co-researchers that they must register with ConnectomeDB
and agree to these terms.

Register and sign the Open Access Data Use Terms at
ConnectomeDB: https://db.humanconnectome.org/
