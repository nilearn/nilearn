
..
    We are putting the title as a raw HTML so that it doesn't appear in
    the contents
    
.. raw:: html

    <h1>nisl: machine learning for Neuro Imaging in Python</h1>
    <style type="text/css">
    p {
        margin: 7px 0 7px 0 ;
    }
    span.linkdescr a {
        color:  #3E4349 ;
    }
    </style>

..  
   Here we are building a banner: a javascript selects randomly 4 images in 
   the list

.. only:: html

    .. |banner1| image:: auto_examples/images/plot_haxby_visualisation_1.png
       :height: 140
       :target: auto_examples/plot_test.html

    .. |center-div| raw:: html

        <div style="text-align: center; margin: -7px 0 -10px 0;" id="banner">

    .. |end-div| raw:: html

        </div>

        <SCRIPT>
        // Function to select 4 imgs in random order from a div
        function shuffle(e) {       // pass the divs to the function
          var replace = $('<div>');
          var size = 4;
          var num_choices = e.size();

          while (size >= 1) {
            var rand = Math.floor(Math.random() * num_choices);
            var temp = e.get(rand);      // grab a random div from our set
            replace.append(temp);        // add the selected div to our new set
            e = e.not(temp); // remove our selected div from the main set
            size--;
            num_choices--;
          }
          $('#banner').html(replace.html() ); // update our container div 
                                              // with the new, randomized divs
        }
        shuffle ($('#banner a.external'));
        </SCRIPT>

    |center-div| |banner1| |end-div|


**License:** Open source, commercially usable: **BSD license** (3 clause)

.. include:: includes/big_toc_css.rst

Documentation for nisl **version** |release|. For other versions and
printable format, see :ref:`documentation_resources`.

User Guide
==========

.. toctree::
   :maxdepth: 2

   tutorial/index.rst

Example Gallery
===============

.. toctree::
   :maxdepth: 2

   auto_examples/index


.. toctree::
   :hidden:

..   support
..   whats_new
