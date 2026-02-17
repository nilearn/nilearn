"""
Copyright (c) 2013, Dave Mankoff
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Dave Mankoff nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL DAVE MANKOFF BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import unicode_literals
import unittest

import htmlmin
from htmlmin.decorator import htmlmin as htmlmindecorator
from htmlmin.middleware import HTMLMinMiddleware

from . import test_escape

MINIFY_FUNCTION_TEXTS = {
  'simple_text': (
    '  a  b',
    ' a b'
  ),
  'long_text': (
    '''When doing     test-driven development, or
    running automated builds that need testing before they are  deployed
\t\t for downloading or use, it's often useful to be able to run a project's
unit tests without actually deploying the project anywhere.\r\n\r\n\n\r\rThe
test command runs project's unit tests without actually deploying it, by
    temporarily putting the project's source on sys.path, after first running
     build_ext -i to ensure that any C extensions are built.

    ''',
    ("When doing test-driven development, or running automated "
      "builds that need testing before they are deployed for "
      "downloading or use, it's often useful to be able to run a "
      "project's unit tests without actually deploying the project "
      "anywhere. The test command runs project's unit tests without "
      "actually deploying it, by temporarily putting the project's "
      "source on sys.path, after first running build_ext -i to "
      "ensure that any C extensions are built. ")
  ),
  'simple_html': (
    (' <body> <b>  a <i pre>b  </i>'  # <b> is not closed
     '<pre>   x </pre> <textarea>   Y  </textarea></body> '),
    (' <body> <b> a <i>b  </i>'
     '<pre>   x </pre> <textarea>   Y  </textarea></body> ')
  ),
  'with_doctype': (
    '\n\n<!DOCTYPE html>\n\n<body>   X   Y   </body>',
    '<!DOCTYPE html><body> X Y </body>'
  ),
  'dangling_tag': (
    "<p>first page",
    "<p>first page"
    ),
  'dangling_tag_followup': (
    "<html><p>next page",
    "<html><p>next page"
    )
}

FEATURES_TEXTS = {
  'remove_quotes': (
    '<body  >  <div id="x" style="   abc " data-a=b></div></  body>  ',
    '<body> <div id=x style="   abc " data-a=b></div></body> ',
  ),
  'remove_quotes_drop_trailing_slash': (
    '<div x="x/" y="y/"></div>',
    # Note: According to https://github.com/mankyd/htmlmin/pull/12 older version
    # of WebKit would erroneously interpret "<... y=y/>" as self-closing tag.
    '<div x=x/ y=y/ ></div>',
  ),
  'remove_quotes_keep_space_before_slash': (
    '<foo x="x/"/>',
    '<foo x=x/ />',  # NOTE: Space added so self-closing tag is parsed as such.
  ),
  'remove_single_quotes': (
    '<body><div thing=\'what\'></div></body> ',
    '<body><div thing=what></div></body> ',
  ),
  'keep_nested_single_quotes': (
    '<body><div thing="wh\'at"></div></body> ',
    '<body><div thing="wh\'at"></div></body> ',
  ),
  'remove_tag_name_whitespace': (
    '<body  >  <br   />  <textarea  >   </ textarea  ></  body>  ',
    '<body> <br> <textarea>   </textarea></body> '
  ),
  'no_reduce_empty_attributes': (
    '<body><img src="/x.png" alt="" /></body>',
    '<body><img src=/x.png alt=""></body>',
  ),
  'no_reduce_empty_attributes_keep_quotes': (
    '<body><img src="/x.png" alt="" /></body>',
    '<body><img src="/x.png" alt=""></body>',
  ),
  'reduce_empty_attributes': (
    '<body><img src="/x.png" alt="" /></body>',
    '<body><img src=/x.png alt></body>',
  ),
  'keep_boolean_attributes': (
    '<body><input id="x" disabled="disabled"></body>',
    '<body><input id=x disabled=disabled></body>',
  ),
  'reduce_boolean_attributes': (
    '<body><input id="x" disabled="disabled"></body>',
    '<body><input id=x disabled></body>',
  ),
  'remove_close_tags': (
    '<body><br></br><br pre>   x   </br> </body>',
    '<body><br><br>   x   </body>',
  ),
  'remove_comments': (
    '<body> this text should <!-- X --> have <!----> comments removed</body>',
    '<body> this text should have comments removed</body>',
  ),
  'keep_comments': (
    '<body> this text should <!--! not --> have comments removed</body>',
    '<body> this text should <!-- not --> have comments removed</body>',
  ),
  'keep_empty_comments': (
    '<body> this text should not<!----> have empty comments removed</body>',
    '<body> this text should not<!----> have empty comments removed</body>',
  ),
  'keep_conditional_comments': (
    '<body>keep IE conditional styles <!--[if IE8]><style>h1 {color: red;}</style><![endif]--></body>',
    '<body>keep IE conditional styles <!--[if IE8]><style>h1 {color: red;}</style><![endif]--></body>',
  ),
  'remove_nonconditional_comments': (
    '<body>remove other [if] things <!-- so [if IE8]--><style>h1 {color: red;}</style><!--[endif]--></body>',
    '<body>remove other [if] things <style>h1 {color: red;}</style></body>',
  ),
  'keep_optional_attribute_quotes': (
    '<img width="100" height="50" src="#something" />',
    '<img width="100" height="50" src="#something">',
  ),
  'remove_optional_attribute_quotes': (
    (
      '<td data-text="&lt;script&gt;alert(\'123\');&lt;/script&gt;">',
      '<td data-text="<script>alert(\'123\');</script>">',
    ),
    (
      '<td data-text="&lt;script&gt;alert(123);&lt;/script&gt;">',
      '<td data-text="<script>alert(123);</script>">',
    ),
  ),
  'keep_pre_attribute': (
    '<body>the <strong pre   style="">pre</strong> should stay  </body>',
    '<body>the <strong pre style>pre</strong> should stay </body>',
  ),
  'custom_pre_attribute': (
    '<body>the <strong pre  >   X  </strong><span custom>  Y  </span></body>',
    '<body>the <strong pre> X </strong><span>  Y  </span></body>',
  ),
  'keep_empty': (
    '<body> <div id="x"  >  A </div>  <div id="  y ">  B    </div>  </body>',
    '<body> <div id=x> A </div> <div id="  y "> B </div> </body>',
  ),
  'remove_empty': (
    ('<body>  \n  <div id="x"  >  A </div>\r'
     '<div id="  y ">  B    </div>\r\n  <div> C </div>  <div>D</div> </body>'),
    ('<body><div id=x> A </div>'
     '<div id="  y "> B </div><div> C </div> <div>D</div> </body>'),
  ),
  'remove_all_empty': (
    ('<body>  \n  <div id=x  >  A </div>\r'
     '<div id="  y ">  B    </div>\r\n  <div> C </div>  <div>D</div> </body>'),
    ('<body><div id=x> A </div>'
     '<div id="  y "> B </div><div> C </div><div>D</div></body>'),
  ),
  'dont_minify_div': (
    '<body>  <div>   X  </div>   </body>',
    '<body> <div>   X  </div> </body>',
  ),
  'minify_pre': (
    '<body>  <pre>   X  </pre>   </body>',
    '<body> <pre> X </pre> </body>',
  ),
  'remove_head_spaces': (
    '<head>  <title>   &#x2603;X  Y  &amp;  Z </title>  </head>',
    '<head><title>&#x2603;X Y &amp; Z</title></head>',
  ),
  'dont_minify_scripts_or_styles': (
    '<body>  <script>   X  </script>  <style>   X</style>   </body>',
    '<body> <script>   X  </script> <style>   X</style> </body>',
  ),
  'remove_close_from_tags': (
    ('<body> <area/> <base/> <br /> <col/><command /><embed /><hr/> <img />'
     '   <input   /> <keygen/> <meta  /><param/><source/><track  /><wbr />'
     '  </body>'),
    ('<body> <area> <base> <br> <col><command><embed><hr> <img>'
     ' <input> <keygen> <meta><param><source><track><wbr>'
     ' </body>'),
  ),
  'remove_space_from_self_closed_tags': (
    '<body>    <y />   <x    /></body>',
    '<body> <y/> <x/></body>',
  ),
  'remove_redundant_lang_0': (
    ('<html><body lang=en><p lang=en>This is an example.'
     '<p lang=pl>I po polsku <span lang=el>and more English</span>.'),
    ('<html><body lang=en><p>This is an example.'
     '<p lang=pl>I po polsku <span lang=el>and more English</span>.'),
  ),
  'convert_charrefs': (
    '<input value="&#34;&#39;&#39;&#39;&lt;&#46;&pi;&gt; &#34;">',
     u'<input value="&#34;\'\'\'<.\u03c0> &#34;">',
  ),
  'convert_charrefs_false': (
    '<input value="&#34;&#39;&#39;&#39;&lt;&#46;&pi;&gt; &#34;">',
    '<input value="&#34;&#39;&#39;&#39;&lt;&#46;&pi;&gt; &#34;">',
  ),
  'dont_convert_pre_attr': (
    '<input pre-value="&#34;&#39;&#39;&#39;&lt;&#46;&pi;&gt; &#34;">',
    '<input value=&#34;&#39;&#39;&#39;&lt;&#46;&pi;&gt; &#34;>',
  ),
}

SELF_CLOSE_TEXTS = {
  'p_self_close': (
    '<body>  <p pre  >  X  <p>  Y  ',
    '<body> <p>  X  <p> Y ',
  ),
  'li_self_close': (
    '<body> <ul>  <li pre  >  X  <li>  Y  <li pre>  Z</ul>   Q',
    '<body> <ul> <li>  X  <li> Y <li>  Z</ul> Q',
  ),
  'dt_self_close': (
    '<body> <dl>  <dt pre  >  X  <dt>  Y  <dt pre>  Z</dt></dl>   Q',
    '<body> <dl> <dt>  X  <dt> Y <dt>  Z</dt></dl> Q',
  ),
  'dd_self_close': (
    '<body> <dl>  <dd pre  >  X  <dd>  Y  <dd pre>  Z</dl>   Q',
    '<body> <dl> <dd>  X  <dd> Y <dd>  Z</dl> Q',
  ),
  'optgroup_self_close': (
    ('<body>   <select  a >  <optgroup pre>   <option>   X</option>   '
     '<optgroup>   <option>   Y</option> </optgroup> </select>   </body>'),
    ('<body> <select a> <optgroup>   <option>   X</option>   '
     '<optgroup> <option> Y</option> </optgroup> </select> </body>'),
  ),
  'option_self_close': (
    ('<body>   <select  a >  <option pre  >   X    '
     '<option> Y    </option></select>   </body>'),
    ('<body> <select a> <option>   X    '
     '<option> Y </option></select> </body>'),
  ),
  'colgroup_self_close': (
    '<body>  <table>  <colgroup pre>   </table></body>',
    '<body> <table> <colgroup>   </table></body>',
  ),
  'tbody_self_close': (
    ('<body>  <table>   <tbody pre>  <tr>  <td> X  </td></tr>  \n'
     '\n  <tbody>   <tr>   <td>Y   </td></tr>\n\n\n   </body>'),
    ('<body> <table> <tbody>  <tr>  <td> X  </td></tr>  \n'
     '\n  <tbody> <tr> <td>Y </td></tr> </body>'),
  ),
  'thead_self_close': (
    ('<body>  <table>   <thead pre>  <tr>  <td> X  </td></tr>  '
     '  <tbody>   <tr>   <td>Y   </td></tr> </body>'),
    ('<body> <table> <thead>  <tr>  <td> X  </td></tr>  '
     '  <tbody> <tr> <td>Y </td></tr> </body>'),
  ),
  'tfoot_self_close': (
    ('<body>  <table>   <tfoot pre>  <tr>  <td> X  </td></tr>  '
     '  <tbody>   <tr>   <td>Y   </td></tr> </body>'),
    ('<body> <table> <tfoot>  <tr>  <td> X  </td></tr>  '
     '  <tbody> <tr> <td>Y </td></tr> </body>'),
  ),
  'tr_self_close': (
    ('<body>  <table>   <thead>  <tr pre >  <td> X  </td>  '
     '    <tr>   <td>Y   </td> </body>'),
    ('<body> <table> <thead> <tr>  <td> X  </td>  '
     '    <tr> <td>Y </td> </body>'),
  ),
  'td_self_close': (
    '<body>  <table>   <thead>  <tr>  <td  pre> X       <td>Y    </body>',
    '<body> <table> <thead> <tr> <td> X       <td>Y </body>',
  ),
  'th_self_close': (
    '<body>  <table>   <thead>  <tr>  <th pre> X        <th>Y    </body>',
    '<body> <table> <thead> <tr> <th> X        <th>Y </body>',
  ),
  'a_p_interaction': ( # the 'pre' functionality continues after the </a>
    '<body><a>   <p pre>  X  </a>    <p>   Y</body>',
    '<body><a> <p>  X  </a>    <p> Y</body>',
  ),
}

SELF_OPENING_TEXTS = {
  'html_closed_no_open': (
    '<head></head><body>  X  </body></html>',
    '<head></head><body> X </body></html>'
  ),
  'head_closed_no_open': (
    '    </head><body>  X  </body>',
    ' </head><body> X </body>' # TODO: we could theoretically kill that leading
                               # space. See HTMLMinParse.handle_endtag
  ),
  'body_closed_no_open': (
    '   X  </body>',
    ' X </body>'
  ),
  'colgroup_self_open': (
    '<body>  <table>  </colgroup>   </table></body>',
    '<body> <table> </colgroup> </table></body>',
  ),
  'tbody_self_open': (
    '<body>  <table>  </tbody>   </table></body>',
    '<body> <table> </tbody> </table></body>',
  ),
  'p_closed_no_open': ( # this isn't valid html, but its worth accounting for
    '<body><div pre>   X  </p>   </div><div>    Y   </p>  </div></body>',
    '<body><div>   X  </p>   </div><div> Y </p> </div></body>',
  ),
}

class HTMLMinTestMeta(type):
  def __new__(cls, name, bases, dct):
    def make_test(text):
      def inner_test(self):
        self.assertEqual(self.minify(text[0]), text[1])
      return inner_test

    for k, v in dct.get('__reference_texts__',{}).items():
      if 'test_'+k not in dct:
        dct['test_'+k] = make_test(v)
    return type.__new__(cls, str(name), bases, dct)

class HTMLMinTestCase(
  HTMLMinTestMeta('HTMLMinTestCase', (unittest.TestCase, ), {})):
  def setUp(self):
    self.minify = htmlmin.minify

class TestMinifyFunction(HTMLMinTestCase):
  __reference_texts__ = MINIFY_FUNCTION_TEXTS

  def test_basic_minification_quality(self):
    import codecs
    with codecs.open('htmlmin/tests/large_test.html', encoding='utf-8') as inpf:
      inp = inpf.read()
    out = self.minify(inp)
    self.assertEqual(len(inp) - len(out), 9408)

  def test_high_minification_quality(self):
    import codecs
    with codecs.open('htmlmin/tests/large_test.html', encoding='utf-8') as inpf:
      inp = inpf.read()
    out = self.minify(inp, remove_all_empty_space=True, remove_comments=True)
    self.assertEqual(len(inp) - len(out), 12518)

class TestMinifierObject(HTMLMinTestCase):
  __reference_texts__ = MINIFY_FUNCTION_TEXTS

  def setUp(self):
    HTMLMinTestCase.setUp(self)
    self.minifier = htmlmin.Minifier()
    self.minify = self.minifier.minify

  def test_reuse(self):
    text = self.__reference_texts__['simple_text']
    self.assertEqual(self.minify(text[0]), text[1])
    self.assertEqual(self.minify(text[0]), text[1])

  def test_dangling_tag(self):
    dangling_tag = self.__reference_texts__['dangling_tag']
    dangling_tag_followup = self.__reference_texts__['dangling_tag_followup']
    self.assertEqual(self.minify(dangling_tag[0]), dangling_tag[1])
    self.assertEqual(self.minify(dangling_tag_followup[0]), dangling_tag_followup[1])

  def test_buffered_input(self):
    text = self.__reference_texts__['long_text']
    self.minifier.input(text[0][:len(text[0]) // 2])
    self.minifier.input(text[0][len(text[0]) // 2:])
    self.assertEqual(self.minifier.finalize(), text[1])

  
class TestMinifyFeatures(HTMLMinTestCase):
  __reference_texts__ = FEATURES_TEXTS

  def test_remove_comments(self):
    text = self.__reference_texts__['remove_comments']
    self.assertEqual(htmlmin.minify(text[0], remove_comments=True), text[1])

  def test_no_reduce_empty_attributes(self):
    text = self.__reference_texts__['no_reduce_empty_attributes']
    self.assertEqual(htmlmin.minify(text[0], reduce_empty_attributes=False), text[1])

  def test_no_reduce_empty_attributes_keep_quotes(self):
    text = self.__reference_texts__['no_reduce_empty_attributes_keep_quotes']
    self.assertEqual(htmlmin.minify(text[0], reduce_empty_attributes=False, remove_optional_attribute_quotes=False), text[1])

  def test_reduce_empty_attributes(self):
    text = self.__reference_texts__['reduce_empty_attributes']
    self.assertEqual(htmlmin.minify(text[0], reduce_empty_attributes=True), text[1])

  def test_reduce_boolean_attributes(self):
    text = self.__reference_texts__['reduce_boolean_attributes']
    self.assertEqual(htmlmin.minify(text[0], reduce_boolean_attributes=True), text[1])

  def test_keep_comments(self):
    text = self.__reference_texts__['keep_comments']
    self.assertEqual(htmlmin.minify(text[0], remove_comments=True), text[1])

  def test_keep_empty_comments(self):
    text = self.__reference_texts__['keep_empty_comments']
    self.assertEqual(htmlmin.minify(text[0]), text[1])

  def test_keep_conditional_comments(self):
    text = self.__reference_texts__['keep_conditional_comments']
    self.assertEqual(htmlmin.minify(text[0], remove_comments=True), text[1])

  def test_remove_nonconditional_comments(self):
    text = self.__reference_texts__['remove_nonconditional_comments']
    self.assertEqual(htmlmin.minify(text[0], remove_comments=True), text[1])

  def test_keep_optional_attribute_quotes(self):
    text = self.__reference_texts__['keep_optional_attribute_quotes']
    self.assertEqual(htmlmin.minify(text[0], remove_optional_attribute_quotes=False), text[1])

  def test_remove_optional_attribute_quotes(self):
    texts = self.__reference_texts__['remove_optional_attribute_quotes']
    for text in texts:
      self.assertEqual(htmlmin.minify(text[0], remove_optional_attribute_quotes=True), text[1])

  def test_keep_pre_attribute(self):
    text = self.__reference_texts__['keep_pre_attribute']
    self.assertEqual(htmlmin.minify(text[0], keep_pre=True), text[1])

  def test_custom_pre_attribute(self):
    text = self.__reference_texts__['custom_pre_attribute']
    self.assertEqual(htmlmin.minify(text[0], pre_attr='custom'), text[1])

  def test_keep_empty(self):
    text = self.__reference_texts__['keep_empty']
    self.assertEqual(htmlmin.minify(text[0]), text[1])

  def test_remove_empty(self):
    text = self.__reference_texts__['remove_empty']
    self.assertEqual(htmlmin.minify(text[0], remove_empty_space=True), text[1])

  def test_remove_all_empty(self):
    text = self.__reference_texts__['remove_all_empty']
    self.assertEqual(htmlmin.minify(text[0], remove_all_empty_space=True),
                     text[1])

  def test_dont_minify_div(self):
    text = self.__reference_texts__['dont_minify_div']
    self.assertEqual(htmlmin.minify(text[0], pre_tags=('div',)), text[1])

  def test_minify_pre(self):
    text = self.__reference_texts__['minify_pre']
    self.assertEqual(htmlmin.minify(text[0], pre_tags=('div',)), text[1])

  def test_remove_head_spaces(self):
    text = self.__reference_texts__['remove_head_spaces']
    self.assertEqual(htmlmin.minify(text[0]), text[1])

  def test_dont_minify_scripts_or_styles(self):
    text = self.__reference_texts__['dont_minify_scripts_or_styles']
    self.assertEqual(htmlmin.minify(text[0], pre_tags=[]), text[1])

  def test_convert_charrefs_false(self):
    text = self.__reference_texts__['convert_charrefs_false']
    self.assertEqual(htmlmin.minify(text[0], convert_charrefs=False), text[1])


class TestSelfClosingTags(HTMLMinTestCase):
  __reference_texts__ = SELF_CLOSE_TEXTS

class TestSelfOpeningTags(HTMLMinTestCase):
  __reference_texts__ = SELF_OPENING_TEXTS

class TestDecorator(HTMLMinTestCase):
  def test_direct_decorator(self):
    @htmlmindecorator
    def directly_decorated():
      return '   X   Y   '

    self.assertEqual(' X Y ', directly_decorated())

  def test_options_decorator(self):
    @htmlmindecorator(remove_comments=True)
    def directly_decorated():
      return '   X <!-- Removed -->  Y   '

    self.assertEqual(' X Y ', directly_decorated())

class TestMiddleware(HTMLMinTestCase):
  def setUp(self):
    HTMLMinTestCase.setUp(self)
    def wsgi_app(environ, start_response):
      start_response(environ['status'], environ['headers'])
      yield environ['content']

    self.wsgi_app = wsgi_app

  def call_app(self, app, status, headers, content):
    response_status = []  # these need to be mutable so that they can be changed
    response_headers = [] # within our inner function.
    def start_response(status, headers, exc_info=None):
      response_status.append(status)
      response_headers.append(headers)
    response_body = ''.join(app({'status': status,
                                 'content': content,
                                 'headers': headers},
                                start_response))
    return response_status[0], response_headers[0], response_body

  def test_middlware(self):
    app = HTMLMinMiddleware(self.wsgi_app)
    status, headers, body = self.call_app(
      app, '200 OK', (('Content-Type', 'text/html'),),
      '    X    Y   ')
    self.assertEqual(body, ' X Y ')

  def test_middlware_minifier_options(self):
    app = HTMLMinMiddleware(self.wsgi_app, remove_comments=True)
    status, headers, body = self.call_app(
      app, '200 OK', (('Content-Type', 'text/html'),),
      '    X    Y   <!-- Z -->')
    self.assertEqual(body, ' X Y ')

  def test_middlware_off_by_default(self):
    app = HTMLMinMiddleware(self.wsgi_app, by_default=False)
    status, headers, body = self.call_app(
      app, '200 OK', (('Content-Type', 'text/html'),),
      '    X    Y   ')
    self.assertEqual(body, '    X    Y   ')

  def test_middlware_on_by_header(self):
    app = HTMLMinMiddleware(self.wsgi_app, by_default=False)
    status, headers, body = self.call_app(
      app, '200 OK', (
        ('Content-Type', 'text/html'),
        ('X-HTML-Min-Enable', 'True'),
        ),
      '    X    Y   ')
    self.assertEqual(body, ' X Y ')

  def test_middlware_off_by_header(self):
    app = HTMLMinMiddleware(self.wsgi_app)
    status, headers, body = self.call_app(
      app, '200 OK', (
        ('Content-Type', 'text/html'),
        ('X-HTML-Min-Enable', 'False'),
        ),
      '    X    Y   ')
    self.assertEqual(body, '    X    Y   ')

  def test_middlware_remove_header(self):
    app = HTMLMinMiddleware(self.wsgi_app)
    status, headers, body = self.call_app(
      app, '200 OK', (
        ('Content-Type', 'text/html'),
        ('X-HTML-Min-Enable', 'False'),
        ),
      '    X    Y   ')
    self.assertFalse(any((h == 'X-HTML-Min-Enable' for h, v in headers)))

  def test_middlware_keep_header(self):
    app = HTMLMinMiddleware(self.wsgi_app, keep_header=True)
    status, headers, body = self.call_app(
      app, '200 OK', [
        ('Content-Type', 'text/html'),
        ('X-HTML-Min-Enable', 'False'),
        ],
      '    X    Y   ')
    self.assertTrue(any((h == 'X-HTML-Min-Enable' for h, v in headers)))

def suite():
    minify_function_suite = unittest.TestLoader().\
        loadTestsFromTestCase(TestMinifyFunction)
    minifier_object_suite = unittest.TestLoader().\
        loadTestsFromTestCase(TestMinifierObject)
    minify_features_suite = unittest.TestLoader().\
        loadTestsFromTestCase(TestMinifyFeatures)
    self_closing_tags_suite = unittest.TestLoader().\
        loadTestsFromTestCase(TestSelfClosingTags)
    self_opening_tags_suite = unittest.TestLoader().\
        loadTestsFromTestCase(TestSelfOpeningTags)
    decorator_suite = unittest.TestLoader().\
        loadTestsFromTestCase(TestDecorator)
    middleware_suite = unittest.TestLoader().\
        loadTestsFromTestCase(TestMiddleware)
    return unittest.TestSuite([
        minify_function_suite,
        minifier_object_suite,
        minify_features_suite,
        self_closing_tags_suite,
        self_opening_tags_suite,
        decorator_suite,
        middleware_suite,
        test_escape.suite(),
        ])

if __name__ == '__main__':
  unittest.main()
