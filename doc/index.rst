
..
    We are putting the title as a raw HTML so that it doesn't appear in
    the contents
    
.. raw:: html

    <h1>Machine learning for NeuroImaging in Python</h1>
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

	//Function to make the index toctree collapsible
	$(function () {
            $('.toctree-l2')
                .click(function(event){
                    if (event.target.tagName.toLowerCase() != "a") {
		        if ($(this).children('ul').length > 0) {
                            $(this).css('list-style-image',
                            (!$(this).children('ul').is(':hidden')) ? 'url(_static/plusBoxHighlight.png)' : 'url(_static/minBoxHighlight.png)');
                            $(this).children('ul').toggle();
                        }
                        var bodywrapper = $('.bodywrapper');
                        var sidebarbutton = $('#sidebarbutton');
                        sidebarbutton.height(bodywrapper.height());
                        return true; //Makes links clickable
                    }
		})
		.mousedown(function(event){ return false; }) //Firefox highlighting fix
                .css({cursor:'pointer', 'list-style-image':'url(_static/plusBox.png)'})
                .children('ul').hide();
            $('ul li ul li:not(:has(ul))').css({cursor:'default', 'list-style-image':'url(_static/noneBox.png)'});
	    $('.toctree-l3').css({cursor:'default', 'list-style-image':'url(_static/noneBox.png)'});

	    $('.toctree-l2').hover(
	        function () {
		    if ($(this).children('ul').length > 0) {
		        $(this).css('background-color', '#D0D0D0').children('ul').css('background-color', '#F0F0F0');
		        $(this).css('list-style-image',
                            (!$(this).children('ul').is(':hidden')) ? 'url(_static/minBoxHighlight.png)' : 'url(_static/plusBoxHighlight.png)');
		    }
		    else {
		        $(this).css('background-color', '#F9F9F9');
		    }
                },
                function () {
                    $(this).css('background-color', 'white').children('ul').css('background-color', 'white');
		    if ($(this).children('ul').length > 0) {
		        $(this).css('list-style-image',
                            (!$(this).children('ul').is(':hidden')) ? 'url(_static/minBox.png)' : 'url(_static/plusBox.png)');
		    }
                }
            );

            var bodywrapper = $('.bodywrapper');
            var sidebarbutton = $('#sidebarbutton');
            sidebarbutton.height(bodywrapper.height());

	});

        </SCRIPT>

    |center-div| |banner1| |banner2| |banner3| |banner4| |banner5| |end-div|

.. only:: html

 .. sidebar:: Download

    * `PDF, 2 pages per side <./_downloads/nisl.pdf>`_

    * `HTML and example files <https://github.com/nisl/nisl.github.com/zipball/master>`_

    * `Source code (github) <https://github.com/nisl/tutorial>`_


 .. topic:: Learn machine learning for fMRI

   This document compiles examples and tutorials to learn to apply
   machine learning to fMRI using Python and the `scikit-learn
   <http://scikit-learn.org>`__. It requires `nibabel
   <htpp://nipy.org/nibabel>`__ and the `scikit-learn
   <http://scikit-learn.org>`__.


.. include:: includes/big_toc_css.rst

Tutorial topics
================

.. toctree::
   :numbered:

   introduction.rst
   getting_started.rst
   supervised_learning.rst
   unsupervised_learning.rst
   dataset_manipulation.rst
   Reference <modules/classes.rst>

.. only:: html

   auto_examples/index


..
 FIXME: I need the link below to make sure the banner gets copied to the
 target directory.


.. only:: html

 .. raw:: html

   <div style='visibility: hidden ; height=0'>

 :download:`nisl.pdf`

 .. raw:: html

   </div>

