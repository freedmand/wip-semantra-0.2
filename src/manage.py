#!/usr/bin/env python

import os


def init_django():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
    import django
    from django.conf import settings

    if settings.configured:
        return

    django.setup()


if __name__ == "__main__":
    from django.core.management import execute_from_command_line

    init_django()
    execute_from_command_line()
