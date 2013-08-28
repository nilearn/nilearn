
..
    We are putting the title as a raw HTML so that it doesn't appear in
    the contents

.. raw:: html

    <style type="text/css">
    p {
	font-size: 18px;
	line-height: 2em;
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
       :height: 130
       :target: auto_examples/plot_haxby_decoding.html

    .. |banner2| image:: auto_examples/images/plot_simulated_data_3.png
       :height: 80
       :target: auto_examples/plot_simulated_data.html

    .. |banner3| image:: auto_examples/images/plot_rest_clustering_1.png
       :height: 130
       :target: auto_examples/plot_rest_clustering.html

    .. |banner4| image:: auto_examples/images/plot_ica_resting_state_1.png
       :height: 130
       :target: auto_examples/plot_ica_resting_state.html

    .. |banner5| image:: auto_examples/images/plot_haxby_searchlight_1.png
       :height: 130
       :target: auto_examples/plot_haxby_searchlight.html


    .. |center-div| raw:: html

        <div style="text-align: center; vertical-align: middle; margin: 40px;" id="banner" class="banner">

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

    |center-div| |banner1| |banner2| |banner3| |banner4| |banner5| |end-div|



Machine Learning for Neuro-Imaging
----------------------------------

   NiLearn is a software package to facilitate the use of statistical learning
   on NeuroImaging data.

   It leverages the `scikit-learn <http://scikit-learn.org>`__ Python toolbox
   for multivariate statistics with applications such as predictive modelling,
   classification, decoding, or connectivity analysis.

   It requires only
   `nibabel <htpp://nipy.org/nibabel>`__ and `scikit-learn
   <http://scikit-learn.org>`__.


.. sidebar:: Download
   :class: green

   * `Source code (github) <https://github.com/nilearn/nilearn>`_

.. warning::

   NiLearn is still an unreleased package in early development stages.

..
 FIXME: I need the link below to make sure the banner gets copied to the
 target directory.


.. raw:: html

   </div>

