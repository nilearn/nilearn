
..
    We are putting the title as a raw HTML so that it doesn't appear in
    the contents
    
.. raw:: html

    <h1>Machine learning for Neuro Imaging in Python</h1>
    <style type="text/css">
    p {
        margin: 7px 0 7px 0 ;
    }
    span.linkdescr a {
        color:  #3E4349 ;
    }
    div.banner img {
        vertical-align: middle;
    }
    </style>

..  
   Here we are building a banner: a javascript selects randomly 4 images in 
   the list

.. only:: html

    .. |banner1| image:: auto_examples/images/plot_haxby_decoding_1.png
       :height: 140
       :target: auto_examples/plot_haxby_decoding.html

    .. |banner2| image:: auto_examples/images/plot_simulated_data_3.png
       :height: 100
       :target: auto_examples/plot_simulated_data.html

    .. |center-div| raw:: html

        <div style="text-align: center; vertical-align: middle; margin: -7px 0 -10px 0;" id="banner" class="banner">

    .. |end-div| raw:: html

        </div>

        <SCRIPT>
        // Function to select 4 imgs in random order from a div
        function shuffle(e) {       // pass the divs to the function
          var replace = $('<div>');
          var size = 4;
          var num_choices = e.size();

          while (size >= 1 && num_choices >= 1) {
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

    |center-div| |banner1| |banner2| |end-div|

.. topic:: Learn machine learning for fMRI

   This document compiles examples and tutorials to learn to apply
   machine learning to fMRI using Python and the `scikit-learn
   <http://scikit-learn.org>`__. It requires `nibabel
   <htpp://nipy.org/nibabel>`__ and the `scikit-learn
   <http://scikit-learn.org>`__.


.. include:: includes/big_toc_css.rst

.. toctree::
   :maxdepth: 2
   :numbered:

   introduction.rst
   haxby_decoding.rst
   auto_examples/index



