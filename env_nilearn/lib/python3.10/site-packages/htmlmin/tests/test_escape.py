"""
Copyright (c) 2015, Dave Mankoff
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

from htmlmin import escape

class TestEscapeAttributes(unittest.TestCase):
  def assertQuotes(self, value, expected, quotes):
    result = escape.escape_attr_value(value)
    self.assertEqual((expected, quotes), result)

  def assertNoQuotes(self, value, expected):
    self.assertQuotes(value, expected, escape.NO_QUOTES)

  def assertSingleQuote(self, value, expected):
    self.assertQuotes(value, expected, escape.SINGLE_QUOTE)

  def assertDoubleQuote(self, value, expected):
    self.assertQuotes(value, expected, escape.DOUBLE_QUOTE)

  def test_simple_attribute(self):
    self.assertNoQuotes('foobar', 'foobar')

  def test_double_quote(self):
    self.assertDoubleQuote("foo'bar", "foo'bar")

  def test_single_quote(self):
    self.assertSingleQuote('foo"bar', 'foo"bar')

  def test_quote_space(self):
    self.assertDoubleQuote("foo bar", "foo bar")
    self.assertDoubleQuote(" foobar", " foobar")
    self.assertDoubleQuote("foobar ", "foobar ")
    self.assertDoubleQuote(" foobar ", " foobar ")
    self.assertDoubleQuote("", "")

  def test_quote_unsafe_chars(self):
    self.assertDoubleQuote("width=device-width", "width=device-width")
    self.assertDoubleQuote("<sinister-brackets>", "<sinister-brackets>")
    self.assertDoubleQuote("`", "`")

  def test_force_double_quote(self):
    result = escape.escape_attr_value("foobar", double_quote=True)
    self.assertEqual('foobar', result[0])
    self.assertEqual(escape.DOUBLE_QUOTE, result[1])

  def test_both_quotes(self):
    self.assertDoubleQuote("foo'\"bar", "foo'&#34;bar")
    self.assertDoubleQuote("foo''\"bar", "foo''&#34;bar")
    self.assertSingleQuote("foo'\"\"bar", 'foo&#39;""bar')

  def test_ampersand_char_ref(self):
    self.assertNoQuotes('foo&bar', 'foo&amp;bar')

  def test_multi_ampersand_char_ref(self):
    self.assertNoQuotes('foo&&bar', 'foo&&amp;bar')
    self.assertNoQuotes('foo&&&bar', 'foo&&&amp;bar')
    self.assertNoQuotes('foo&&&&bar', 'foo&&&&amp;bar')
    self.assertNoQuotes('foo&&bar&&baz', 'foo&&amp;bar&&amp;baz')

  def test_ampersand_decimal(self):
    self.assertNoQuotes('foo&#34', 'foo&amp;#34')

  def test_ampersand_non_decimal(self):
    self.assertNoQuotes('foo&#ff', 'foo&#ff')

  def test_ampersand_hex(self):
    self.assertNoQuotes('foo&#x34f', 'foo&amp;#x34f')

  def test_ampersand_nonhex(self):
    self.assertNoQuotes('foo&#xz34f', 'foo&#xz34f')

  def test_proper_char_refs(self):
    self.assertNoQuotes('&pi;&#34;&#x34;', '&pi;&#34;&#x34;')

def suite():
  return unittest.TestLoader().loadTestsFromTestCase(TestEscapeAttributes)
