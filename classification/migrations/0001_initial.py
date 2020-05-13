# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import classification.models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataSet',
            fields=[
                ('id', models.AutoField(serialize=False, auto_created=True, verbose_name='ID', primary_key=True)),
                ('title', models.CharField(max_length=20)),
                ('train', models.FileField(upload_to='data')),
                ('test', models.FileField(blank=True, upload_to='data')),
                ('val', models.FileField(blank=True, upload_to='data')),
                ('categories', classification.models.ListField(null=True)),
                ('vocab_filename', models.CharField(blank=True, max_length=200)),
                ('vector_word_filename', models.CharField(blank=True, max_length=200)),
                ('vector_word_npz', models.CharField(blank=True, max_length=200)),
                ('desc', models.TextField(null=True)),
                ('is_hide', models.BooleanField(default=False)),
                ('is_del', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='DataSetModel',
            fields=[
                ('id', models.AutoField(serialize=False, auto_created=True, verbose_name='ID', primary_key=True)),
                ('dataset', models.CharField(max_length=20)),
                ('method', models.CharField(max_length=20)),
                ('title', models.CharField(max_length=50)),
                ('model', models.FilePathField()),
                ('is_del', models.BooleanField(default=False)),
            ],
        ),
    ]
