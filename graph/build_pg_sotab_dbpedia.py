import os
import csv
import pprint
import neo4j
from neo4j import GraphDatabase

from utils.file_registry import get_path

def collect_category(file_path, result, type_or_property):

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[0] == 'table_name':
                continue

            tbname = "_".join(row[0].split('_')[:-1])
            col_idx = row[1] if type_or_property == "type" else row[2]
            category = row[2] if type_or_property == "type" else row[3]

            if tbname not in result:
                result[tbname] = dict()

            if col_idx not in result[tbname]:
                result[tbname][col_idx] = dict()

            result[tbname][col_idx][type_or_property] = category

    # print("New tables are taking into account: {}".format(cnt))
    return result

def type_property_info_by_shared_column(file_paths):
    table_dict = {}
    for file_path in file_paths:
        if 'CTA' in file_path:
            table_dict = collect_category(file_path, table_dict, 'type')
        else:
            table_dict = collect_category(file_path, table_dict, 'property')

    type2property = dict()
    property2type = dict()
    for tb, tb_value in table_dict.items():
        for col_idx, col_category in tb_value.items():
            if 'type' in col_category and 'property' in col_category:
                if col_category['type'] not in type2property:
                    type2property[col_category['type']] = dict()
                    type2property[col_category['type']][col_category['property']] = 1
                else:
                    if col_category['property'] not in type2property[col_category['type']]:
                        type2property[col_category['type']][col_category['property']] = 1
                    else:
                        type2property[col_category['type']][col_category['property']] += 1

                if col_category['property'] not in property2type:
                    property2type[col_category['property']] = dict()
                    property2type[col_category['property']][col_category['type']] = 1
                else:
                    if col_category['type'] not in property2type[col_category['property']]:
                        property2type[col_category['property']][col_category['type']] = 1
                    else:
                        property2type[col_category['property']][col_category['type']] += 1

    # pprint.pprint(property2type)
    # pprint.pprint(type2property)

    return property2type, type2property


def list_property_per_topic(file_paths):
    result = {}
    for file_path in file_paths:
        if 'CTA' in file_path:
            continue

        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:

                if row[0] == 'table_name':
                    continue

                topic = row[0].split('_')[0]
                if topic not in result.keys():
                    result[topic] = {'property': set()}

                catogory = row[3]
                result[topic]['property'].add(catogory)

    # pprint.pprint(result)
    return result

def list_type_per_topic(file_paths):
    result = {}
    for file_path in file_paths:
        if 'CTA' not in file_path:
            continue

        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:

                if row[0] == 'table_name':
                    continue

                topic = row[0].split('_')[0]
                if topic not in result.keys():
                    result[topic] = {'type': set()}

                catogory = row[2]
                result[topic]['type'].add(catogory)

    # pprint.pprint(result)
    return result


def list_subj_candidate_per_topic(file_paths):
    result = {}
    for file_path in file_paths:
        if 'CTA' not in file_path:
            continue
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if row[0] == 'table_name':
                    continue

                topic = row[0].split('_')[0]
                if topic not in result.keys():
                    result[topic] = {'subj': dict()}

                catogory = row[2]
                if int(row[1]) in [0, 1]:
                    if catogory not in result[topic]['subj']:
                        result[topic]['subj'][catogory] = 1
                    else:
                        result[topic]['subj'][catogory] += 1
                else:
                    if catogory not in result[topic]['subj']:
                        result[topic]['subj'][catogory] = 0
                    else:
                        result[topic]['subj'][catogory] += 0

    # pprint.pprint(result)
    return result


def construct_SP(topic2S, topic2P):
    topic2SP = {}
    for topic, value in topic2P.items():
        topic2SP[topic] = [[topic2S[topic]['subj'], P] for P in value['property']]

    # pprint.pprint(topic2SP)
    return topic2SP

def construct_SPO(topic2SP, PO):
    topic2SPO = {}
    for topic, sp_list in topic2SP.items():
        if topic not in topic2SPO:
            topic2SPO[topic] = list()

        for s, p in sp_list:

            if p not in PO:
                PO[p] = ['?' + p.split('/')[-1]]
            for o in PO[p]:
                if o == 'https://dbpedia.org/ontology/dateTime' and p == 'https://dbpedia.org/ontology/time' and topic in [
                    'LocalBusiness', 'Place', 'Restaurant']:
                    continue
                if o == 'https://dbpedia.org/ontology/Image' and p == 'https://dbpedia.org/ontology/image' and topic in [
                    'CreativeWork', 'Event', 'JobPosting', 'MusicAlbum', 'MusicRecording', 'Person', 'Recipe']:
                    continue

                topic2SPO[topic].append([s, p, o])

    # pprint.pprint(topic2SPO)
    return topic2SPO


def construct_SPO_with_misalignment(topic2SPO, topic2S):
    # After observing the output from function "detect_misalignment_*"

    t_dict1_add_p = {
        'https://dbpedia.org/ontology/time': {'Recipe'},
        'https://dbpedia.org/ontology/Person': {'LocalBusiness'},
        'https://dbpedia.org/ontology/Organisation': {'Person', 'LocalBusiness'},
        'https://dbpedia.org/ontology/date': {'JobPosting', 'LocalBusiness'},
        'https://dbpedia.org/ontology/SportsTeam': {'SportsEvent'},
        'https://dbpedia.org/ontology/Identifier': {'Event', 'JobPosting', 'Person', 'Product'},
        'https://dbpedia.org/ontology/Street': {'Event', 'Hotel', 'LocalBusiness', 'Museum', 'Person', 'Place',
                                                'Restaurant'},
        'https://dbpedia.org/ontology/address': {'CreativeWork', 'JobPosting', 'LocalBusiness', 'Product'},
        'https://dbpedia.org/ontology/MusicalArtist': {'Movie'},
        'https://dbpedia.org/ontology/distance': {'Person', 'Product'},
        'https://dbpedia.org/ontology/duration': {'Recipe'},
        'https://dbpedia.org/ontology/boolean': {'Book', 'CreativeWork', 'Event', 'Hotel', 'Restaurant'},
        'https://dbpedia.org/ontology/energy': {'Recipe'},
        'https://dbpedia.org/ontology/mass': {'Recipe'},
        'https://dbpedia.org/ontology/List': {'MusicAlbum', 'Recipe'},
    }
    for o, topic_list in t_dict1_add_p.items():
        for topic in topic_list:
            topic2SPO[topic].append([topic2S[topic]['subj'], '??' + o.split('/')[-1], o])

    for topic in ['JobPosting']:
        topic2SPO[topic].append(
            [topic2S[topic]['subj'], 'https://dbpedia.org/ontology/city', 'https://dbpedia.org/ontology/Locality'])
        topic2SPO[topic].append(
            [topic2S[topic]['subj'], 'https://dbpedia.org/ontology/currency', 'https://dbpedia.org/ontology/Currency'])

    for topic in ['Person', 'Event']:
        topic2SPO[topic].append(
            [topic2S[topic]['subj'], 'https://dbpedia.org/ontology/language', 'https://dbpedia.org/ontology/Language'])

    for topic in ['JobPosting', 'Product']:
        topic2SPO[topic].append(
            [topic2S[topic]['subj'], 'https://dbpedia.org/ontology/time', 'https://dbpedia.org/ontology/dateTime'])

    # pprint.pprint(topic2SPO)

    total_collections = set()
    for topic, spo_list in topic2SPO.items():
        for s, p, o in spo_list:
            total_collections.add(s)
            total_collections.add(o)

    # print(len(total_collections))
    # pprint.pprint(total_collections)

    return topic2SPO

def detect_misalignment_property(topic2SPO, file_paths):
    topic2P = list_property_per_topic(file_paths)
    # pprint.pprint(topic2P)

    detect_dict = {}
    for topic, spo_list in topic2SPO.items():
        for s, p, o in spo_list:
            if p not in topic2P[topic]['property']:
                if topic not in detect_dict:
                    detect_dict[topic] = dict()
                if "added_p" not in detect_dict[topic]:
                    detect_dict[topic]["added_p"] = set()
                if p[0:2] != '??':
                    detect_dict[topic]["added_p"].add(p)

    # pprint.pprint(detect_dict)

    detect_dict = {}
    for topic, p_list in topic2P.items():
        p_collections = set()
        for s, p, o in topic2SPO[topic]:
            p_collections.add(p)
        for p in p_list['property']:
            if p not in p_collections:
                if topic not in detect_dict:
                    detect_dict[topic] = {"missing_p": set()}
                detect_dict[topic]["missing_p"].add(p)

    # pprint.pprint(detect_dict)

    cnt_dict = {}
    total_collections = set()
    for topic, p_list in topic2P.items():
        p_collections = set()
        for s, p, o in topic2SPO[topic]:
            p_collections.add(p)
            total_collections.add(p)
        cnt_dict[topic] = len(p_collections)

    # print(cnt_dict)
    # print({topic:len(value['property']) for topic, value in topic2P.items() })
    # print(total_collections)
    # print(len(total_collections))

def detect_misalignment_type(topic2SPO, file_paths):
    topic2T = list_type_per_topic(file_paths)
    # pprint.pprint(topic2T)
    detect_dict = {}
    for topic, spo_list in topic2SPO.items():
        for s, p, o in spo_list:
            if s not in topic2T[topic]['type']:
                if topic not in detect_dict:
                    detect_dict[topic] = dict()
                if "added_subj" not in detect_dict[topic]:
                    detect_dict[topic]["added_subj"] = set()
                detect_dict[topic]["added_subj"].add(s)

            if o not in topic2T[topic]['type']:
                if topic not in detect_dict:
                    detect_dict[topic] = dict()
                if "added_obj" not in detect_dict[topic]:
                    detect_dict[topic]["added_obj"] = set()
                detect_dict[topic]["added_obj"].add(o)

    # pprint.pprint(detect_dict)

    detect_dict = {}
    for topic, t_list in topic2T.items():
        type_collections = set()
        for s, p, o in topic2SPO[topic]:
            type_collections.add(s)
            type_collections.add(o)
        for t in t_list['type']:
            if t not in type_collections:
                if topic not in detect_dict:
                    detect_dict[topic] = {"missing_type": set()}
                detect_dict[topic]["missing_type"].add(t)

    detect_dict_reversed = {}
    for topic, v in detect_dict.items():
        t_list = v['missing_type']
        for t in t_list:
            if t not in detect_dict_reversed:
                detect_dict_reversed[t] = set()
            detect_dict_reversed[t].add(topic)

    # pprint.pprint(detect_dict)
    # pprint.pprint(detect_dict_reversed)

    cnt_dict = {}
    total_collections = set()
    for topic, t_list in topic2T.items():
        type_collections = set()
        for s, p, o in topic2SPO[topic]:
            type_collections.add(s)
            type_collections.add(o)
            total_collections.add(s)
            total_collections.add(o)
        cnt_dict[topic] = len(type_collections)

    # print(cnt_dict)
    # print({topic:len(value['type']) for topic, value in topic2T.items() })
    # print(total_collections)
    # print(len(total_collections))

def table_gt_info(dataset_path=None):

    node_train_gt_path = get_path(dataset_path, 'CTA_DBP_TRAIN_GT_CSV', 'sotab')
    node_val_gt_path = get_path(dataset_path, 'CTA_DBP_VAL_GT_CSV', 'sotab')
    edge_train_gt_path = get_path(dataset_path, 'CPA_DBP_TRAIN_GT_CSV', 'sotab')
    edge_val_gt_path = get_path(dataset_path, 'CPA_DBP_VAL_GT_CSV', 'sotab')

    file_paths = [node_train_gt_path, node_val_gt_path, edge_train_gt_path, edge_val_gt_path]

    # Step 1: find the main subject candidates based on frequencies
    list_subj_candidate_per_topic(file_paths)

    # Step 2: put the printed dict into a file:topic2type and find out the only relevant subject for each topic based on frequencies manually
    topic2S = {
        'Book': {'subj': 'https://dbpedia.org/ontology/Book'},
        'CreativeWork': {'subj': 'https://dbpedia.org/ontology/WrittenWork'},
        'Event': {'subj': 'https://dbpedia.org/ontology/Event'},
        'Hotel': {'subj': 'https://dbpedia.org/ontology/Hotel'},
        'JobPosting': {'subj': '?JobPosting'},  # subj unavailable
        'LocalBusiness': {'subj': 'https://dbpedia.org/ontology/Company'},
        'Movie': {'subj': 'https://dbpedia.org/ontology/Film'},
        'Museum': {'subj': 'https://dbpedia.org/ontology/Museum'},
        'MusicAlbum': {'subj': 'https://dbpedia.org/ontology/Album'},
        'MusicRecording': {'subj': 'https://dbpedia.org/ontology/Song'},
        'Person': {'subj': 'https://dbpedia.org/ontology/Person'},
        'Place': {'subj': 'https://dbpedia.org/ontology/address'},
        'Product': {'subj': '?Product'},  # subj unavailable
        'Recipe': {'subj': '?Recipe'},  # subj unavailable
        'Restaurant': {'subj': 'https://dbpedia.org/ontology/Restaurant'},
        'SportsEvent': {'subj': 'https://dbpedia.org/ontology/SportsEvent'},
        'TVEpisode': {'subj': 'https://dbpedia.org/ontology/TelevisionEpisode'}
    }

    # Step 3: Merge main subject with all the properties for each topic
    topic2P = list_property_per_topic(file_paths)
    topic2SP = construct_SP(topic2S, topic2P)

    # Step 4 column matching between CTA tables & CPA tables: manual work required
    # property-> type (obejct)
    # (subject/object) type -> property
    type_property_info_by_shared_column(file_paths)

    PO = {
        'https://dbpedia.org/ontology/address': {'https://dbpedia.org/ontology/Place'},
        'https://dbpedia.org/ontology/album': {'https://dbpedia.org/ontology/Album'},
        'https://dbpedia.org/ontology/albumRuntime': {'https://dbpedia.org/ontology/duration'},
        'https://dbpedia.org/ontology/artist': {'https://dbpedia.org/ontology/MusicalArtist'},
        'https://dbpedia.org/ontology/author': { # 'https://dbpedia.org/ontology/Organisation',
                                                'https://dbpedia.org/ontology/Person'},
        'https://dbpedia.org/ontology/birthDate': {'https://dbpedia.org/ontology/date'},
        'https://dbpedia.org/ontology/birthPlace': {'https://dbpedia.org/ontology/address'},
        'https://dbpedia.org/ontology/brand': {'https://dbpedia.org/ontology/brand'},
        'https://dbpedia.org/ontology/category': {'https://dbpedia.org/ontology/Category'},
        'https://dbpedia.org/ontology/city': {'https://dbpedia.org/ontology/Locality'},
        'https://dbpedia.org/ontology/country': {'https://dbpedia.org/ontology/Country'},
        'https://dbpedia.org/ontology/cuisine': {'?cuisine'},
        'https://dbpedia.org/ontology/currency': {'https://dbpedia.org/ontology/Currency'},
        'https://dbpedia.org/ontology/day': {'https://dbpedia.org/ontology/day'},
        'https://dbpedia.org/ontology/deathDate': {'https://dbpedia.org/ontology/date'},
        'https://dbpedia.org/ontology/description': {'?description'},
        'https://dbpedia.org/ontology/director': {'https://dbpedia.org/ontology/Person'},
        'https://dbpedia.org/ontology/employer': {'https://dbpedia.org/ontology/Organisation'},
        'https://dbpedia.org/ontology/endDateTime': {# 'https://dbpedia.org/ontology/date',
                                                     'https://dbpedia.org/ontology/dateTime'},
        'https://dbpedia.org/ontology/episodeNumber': {'?episodeNumber'},
        'https://dbpedia.org/ontology/gender': {'https://dbpedia.org/ontology/gender'},
        'https://dbpedia.org/ontology/genre': {'?genre'},
        'https://dbpedia.org/ontology/image': {# 'https://dbpedia.org/ontology/Image',
                                               'https://dbpedia.org/ontology/Website'},
        'https://dbpedia.org/ontology/isbn': {'https://dbpedia.org/ontology/Identifier'},
        'https://dbpedia.org/ontology/language': {'https://dbpedia.org/ontology/Language'},
        'https://dbpedia.org/ontology/locationName': {'https://dbpedia.org/ontology/address'},
        'https://dbpedia.org/ontology/manufacturer': {'https://dbpedia.org/ontology/Organisation'},
        'https://dbpedia.org/ontology/nationality': {'https://dbpedia.org/ontology/Country'},
        'https://dbpedia.org/ontology/performer': {'https://dbpedia.org/ontology/Organisation'},
        'https://dbpedia.org/ontology/postalCode': {'https://dbpedia.org/ontology/PostalCode'},
        'https://dbpedia.org/ontology/price': {'https://dbpedia.org/ontology/price'},
        'https://dbpedia.org/ontology/productionCompany': {'https://dbpedia.org/ontology/Organisation'},
        'https://dbpedia.org/ontology/publicationDate': { #'https://dbpedia.org/ontology/dateTime',
                                                         'https://dbpedia.org/ontology/date'},
        'https://dbpedia.org/ontology/publisher': {'https://dbpedia.org/ontology/Organisation'},
        # corrected
        'https://dbpedia.org/ontology/rating': {'https://dbpedia.org/ontology/rating'},
        'https://dbpedia.org/ontology/region': {'https://dbpedia.org/ontology/Region'},
        'https://dbpedia.org/ontology/releaseDate': {'https://dbpedia.org/ontology/date'},
        'https://dbpedia.org/ontology/review': {'https://dbpedia.org/ontology/review'},
        'https://dbpedia.org/ontology/starring': {'https://dbpedia.org/ontology/Person'},
        'https://dbpedia.org/ontology/startDateTime': { # 'https://dbpedia.org/ontology/date',
                                                       'https://dbpedia.org/ontology/dateTime'},
        'https://dbpedia.org/ontology/televisionSeries': {'https://dbpedia.org/ontology/TelevisionShow'},
        'https://dbpedia.org/ontology/time': { # 'https://dbpedia.org/ontology/dateTime',
                                              'https://dbpedia.org/ontology/time'},
        'https://dbpedia.org/ontology/title': {'?title'},
        'https://dbpedia.org/ontology/totalTracks': {'?totalTracks'},
        'https://dbpedia.org/ontology/weight': {'https://dbpedia.org/ontology/weight'}}

    # STEP 5 merge
    topic2SPO = construct_SPO(topic2SP, PO)

    # STEP 6 detect if there were 1) missing / 2) new-added / 3) wrongly-added types/properties
    # the information of correct wrongly-added types -> contributes to -> handle special cases with conditions in Step 5 again
    detect_misalignment_type(topic2SPO, file_paths)
    detect_misalignment_property(topic2SPO, file_paths)
    topic2SPO = construct_SPO_with_misalignment(topic2SPO, topic2S)

    SPO = [spo for topic, spo_list in topic2SPO.items() for spo in spo_list]
    # pprint.pprint(SPO)

    return SPO, topic2S

def create_pg_neo4j(SPO):

    # step 1: connect to NEO4J DB
    URI = "neo4j+s://"
    AUTH = ("neo4j", "")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()

    ndata = set( [ spo[0] for spo in SPO ] + [ spo[2] for spo in SPO ] )

    # # step 2: create nodes
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        nmap = {}
        for i, node in enumerate(ndata):

            for symbol in ['?', '+']:
                if symbol in node:
                    nname = node.split(symbol)[-1]
                else:
                    nname = node.split('/')[-1]
            nname = ''.join(filter(str.isalnum, nname)) + '_{}'.format(i)
            nmap[node] = nname

            create_node_query = ("CREATE (node:%s { label: $label, source: $source })" % (nname))
            create_node_parameters = {
                "label": nname,
                "source": "'" + node + "'",
            }
            record = driver.execute_query(
                query_=create_node_query,
                parameters_=create_node_parameters,
                routing_=neo4j.RoutingControl.WRITE,
                database_="neo4j",
            )

    # step 3: create edges
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        emap = {}
        for s, p, o in SPO:

            for symbol in ['?', '+']:
                if symbol in node:
                    ename = p.split(symbol)[-1]
                else:
                    ename = p.split('/')[-1]
            ename =''.join(filter(str.isalnum, ename))
            emap[p] = ename

            create_relationship_query = (
                    "MATCH (node1:{})  ".format(nmap[s]) +
                    "MATCH (node2:{})  ".format(nmap[o]) +
                    "CREATE (node1) -[r:%s { label:\"%s\" }]-> (node2)  " % (ename, p) +
                    "RETURN r"
            )
            print(create_relationship_query)
            record = driver.execute_query(
                query_=create_relationship_query,
                routing_=neo4j.RoutingControl.WRITE,
                database_="neo4j",
            )

def main():
    dataset_path = "/apollo/users/dya/dataset/semtab"
    SPO, topic2S = table_gt_info(dataset_path)
    for spo in SPO:
        print(spo)
    # create_pg_neo4j(SPO)

if __name__ == "__main__":
    main()