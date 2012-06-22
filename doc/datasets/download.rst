.. _datasets:

=========================
Dataset loading utilities
=========================

.. currentmodule:: nisl.datasets

The ``nisl.datasets`` package embeds tools to fetch and load datasets. It comes
with a set of several datasets that can be easily downloaded.


Dataset directory
=================
The purpose of this API is to download datasets, preprocess them if needed, but
do it once and for all, so datasets must be stored somewhere.

There are 3 simple ways to determine where to stock datasets. Here are these
rules ordered by priority (the first rule override the others and so on):

1. the keyword argument *data_dir* can be given ton any dataset fetching
   function to force the storage directory.
2. the environment variable *NISL_DATA*
3. by default, a directory called *nisl_data* is created in the current working
   directory.

Loading a dataset
=================

A generic dataset fetching function is available (*fetch_dataset*). Please see
its documentation to learn how to use it.

If you consider using an online public dataset, do not hesitate to follow the
steps below to create a dataset fetching function for this dataset. Any pull
request is welcome.

Writing your own dataset loading function
=========================================

Writing a dataset fetching function is rather easy if the data do not require
complex preprocessing. Take special care of sharing conditions of the dataset
you want to load. If a registration is required, contact the dataset provider
to know if there is a way to get round it (put data on a public server or
create an account that will be used by the script to download it).

Create your fetching function
-----------------------------

Creating your function is straightforward. Your function have to take at least
two parameters :

- *data_dir* if the user wants to override the download directory
- *force_download* is the user wants to download the data again

You don't have to worry about these parameters. They just have to be passed to
some helper functions. You can obviously add custom parameters to fit your*
needs. For example:

- *session*, *subjects* to load only part of the data (may be useful if the
  dataset is composed of several large files)
- *preprocessing* if you want to provide several preprocessings
- *type* to load either anatomical of functional MRI

With the definition function comes the associated docstring. As well as
parameters definition, any information about data structure and/or paper
references is welcome.

.. literalinclude:: ../../nisl/datasets.py
     :start-after: ### Haxby: function definition
     :end-before: ### Haxby: definition of dataset files

Definition of dataset files
---------------------------

The first step is to define of which files is composed the dataset. A simple
array is enough.

.. literalinclude:: ../../nisl/datasets.py
     :start-after: ### Haxby: definition of dataset files
     :end-before: ### Haxby: load the dataset

Load or download the dataset
----------------------------

Now, we try to load the files specified before. Is they cannot be found, we
will try to download the dataset. All these steps can be done simply thanks
to helper functions :

- *get_dataset* is used to load dataset files. An IOError is raised if all
  dataset files are not present
- *fetch_dataset* given a list of urls, try to download all the files of
  a dataset. Progress information is provided.
- *uncompress_dataset* try to uncompress a dataset in its directory

.. literalinclude:: ../../nisl/datasets.py
     :start-after: ### Haxby: load the dataset
     :end-before: ### Haxby: preprocess data

Preprocessing
-------------

If needed, you can preprocess the dataset. As many datasets are in matlab or
Nifti format, reformatting it in a more user-friendly format (like numpy
arrays) is encouraged.

.. literalinclude:: ../../nisl/datasets.py
     :start-after: ### Haxby: preprocess data
     :end-before: ### Haxby: return data

Return dataset
--------------

A convenient way to return a dataset is to use the *Bunch* structure which
encapsulte it and provide easy access to all data fields.

.. literalinclude:: ../../nisl/datasets.py
     :start-after: ### Haxby: return data
     :end-before: ### Haxby: end
