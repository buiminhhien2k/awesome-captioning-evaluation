"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import re
from urllib.parse import urlparse


# def is_url(url_or_filename):
#     parsed = urlparse(url_or_filename)
#     return parsed.scheme in ("http", "https")

def get_abs_path(rel_path):
    # Gets the absolute path of the directory containing this script
    # 1. Get the directory containing this script (main-dir/models/blip2/common)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up 3 levels to reach 'main-dir'
    main_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

    # 3. Join with the incoming relative path
    return os.path.join(main_dir, rel_path)


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url
