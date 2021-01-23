#!/usr/bin/env python3

from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph, Spacer, Table, Image, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm

import email.message
import mimetypes
import os.path
import shutil
import smtplib
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_list_of_lists(_list, _ncols=2):
    """Convert a list in a list of lists of length _ncols.
    """
    list_of_lists = []
    tmp_list = []
    for ele in _list:
        tmp_list.append(ele)
        if len(tmp_list) % _ncols == 0:
            list_of_lists.append(tmp_list)
            tmp_list = []
    # Check if some elements of _list is remained and eventually add them padding with empty elements
    if tmp_list:
        for empty in range(_ncols - len(tmp_list)):
            tmp_list.append('')
        list_of_lists.append(tmp_list)

    return list_of_lists


def generate_report(_filename, _data_file):
    """Generate a report with the results of the KMeans scanning.
    """
    try:
        title = 'Kmeans Scanning Results'
        data = pd.read_csv(_data_file)
        paragraph_1 = 'KMeans scanning for values of k between {:n} and {:n}.'.format(data.k.describe()['min'], data.k.describe()['max'])
        paragraph_2 = 'Graphical representation of the cluster centers for each value of k.'
        paragraph_3 = 'Tabular representation of the cluster centers for each value of k.'

        # Convert the numerical data to string with proper formatting.
        data_list = data.to_numpy().tolist()
        precision=2
        table_row = '{:n}' + (len(data_list[0]) -1) * ',{{:.{:n}f}}'.format(precision)
        table_data = [data.columns.to_list()]
        table_data.extend([table_row.format(*row).split(',') for row in data_list])
        styles = getSampleStyleSheet()
        report = SimpleDocTemplate(_filename)
        report_title = Paragraph(title, styles["h1"])
        #
        # Paragraph 1
        #
        report_paragraph_1 = Paragraph(paragraph_1, styles["BodyText"])
 
        # Generate a graphical form of the results.
        score_plot = sns.pointplot(data=data, x='k', y='score')
        plt.title('Silhuette scores by k')
        plt.ylabel('')
        #if os.path.exists('tmp_plots'):
        #    shutil.rmtree('tmp_plots')
 
        tmp_plots_dir = 'tmp_plots'
        os.mkdir(tmp_plots_dir)
        score_path = os.path.join(tmp_plots_dir, 'score.png')
        plt.savefig(score_path)
 
        report_image_scan = Image(score_path, width=10*cm, height=10*cm)
        report_image_scan.hAlign='CENTER'
 
        #
        # Paragraph 2
        #
        report_paragraph_2 = Paragraph(paragraph_2, styles["BodyText"])
        # Add the cluster identificaion for each k to the dataframe.
        data['cluster'] = data.groupby('k').cumcount() + 1

        # For each row will be darwn 2 plots. The plots will be generated row by row.
        # The use of facetting here is avoided because it could lead to undesired splitting 
        # of the image throughout the pages.
        data_melted = data.melt(id_vars=['k','cluster'], var_name='features')

        k_list = data_melted.k.unique()
        k_lenght = len(k_list)
        
        if k_lenght == 1:
            # Only one cluster to plot.
            score_plot = sns.pointplot(data=data, x='k', y='score')
            plt.title('Silhuette scores by k')
            plt.ylabel('')
            clusters_path = os.path.join(tmp_plots_dir, 'cluster.png')
            plt.savefig(clasters_path)
 
            report_image_clusters = Image(clusters_path, width=10*cm, height=10*cm)
            report_image_clusters.hAlign='CENTER'
        
        else:
            images_list = []
            for k in data_melted.k.unique():
                sns.pointplot(data=data_melted.loc[data_melted.k == k,:], x='features', y='value', hue='cluster')
                plt.title('k={}'.format(k))
                plt.xticks(rotation=90)
                tmp_cluster_path = os.path.join(tmp_plots_dir, 'clusters_plot_{}.jpg'.format(k))
                plt.savefig(tmp_cluster_path, bbox_inches='tight')
                cluster_image = Image(tmp_cluster_path, width=8*cm, height=8*cm)
                images_list.append(cluster_image)
                plt.clf()

            report_image_clusters = Table(data=make_list_of_lists(images_list), hAlign='CENTER')




        #
        # Paragraph 3
        #
        report_paragraph_3 = Paragraph(paragraph_3, styles["BodyText"])
        # Generate a tablular form of the results.
        table_style = TableStyle(
                      [('GRID', (0,0), (-1,-1), 1, colors.black),
                       ('BOX', (0,0), (-1,0), 2, colors.black),
                       ('BACKGROUND', (0,0), (-1,0), colors.grey),
                       ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                       ('ALIGN', (0,0), (-1,-1), 'CENTER')])
        
        for index, row in data.iterrows():
            # Alternate the background color in the table row to better highlight
            # the centers belonging to the same k value.
            if row['k']%2 == 0:
                table_style.add('BACKGROUND', (0, index+1), (-1, index+1), colors.lightgreen)
        report_table = Table(data=table_data, style=table_style, hAlign="CENTER")
        empty_line = Spacer(1,20)
 
        report.build([report_title, empty_line, report_paragraph_1, empty_line, report_image_scan, 
            empty_line, report_paragraph_2, empty_line, report_image_clusters, empty_line, report_paragraph_3,
            empty_line, report_table])
    finally:
        shutil.rmtree(tmp_plots_dir)


def generate_email(_sender, _recipient, _subject, _body, _attachment_path=None):
    """Creates an email with an optional attachement."""
    # Basic Email formatting
    message = email.message.EmailMessage()
    message["From"] = _sender
    message["To"] = _recipient
    message["Subject"] = _subject
    message.set_content(_body)
    
    if _attachment_path is not None:
        # Process the attachment and add it to the email
        attachment_filename = os.path.basename(_attachment_path)
        mime_type, _ = mimetypes.guess_type(_attachment_path)
        mime_type, mime_subtype = mime_type.split('/', 1)
    
        try:
            with open(_attachment_path, 'rb') as ap:
                message.add_attachment(ap.read(),
                maintype=mime_type,
                subtype=mime_subtype,
                filename=attachment_filename)
        except FileNotFoundError:
            print("File {} does not exist.".format(_attachment_path))
            sys.exit(1)
        
    return message

def send_email(message):
    """Sends the message to the configured SMTP server."""
    mail_server = smtplib.SMTP('localhost')
    mail_server.send_message(message)
    mail_server.quit()

