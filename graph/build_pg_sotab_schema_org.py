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
    # After observing the output from function "detect_misalignment_*"
    # handle added obj/subj in construct SPO func

    topic_bad_cases = {
         'Book': {'added_obj': {'price-priceRange'}},
         'CreativeWork': {'added_obj': {'image-Photograph', 'price-priceRange'}},
         'Event': {'added_obj': {'addressRegion-addressRegion','image-Photograph','price-priceRange'}},
         'Hotel': {'added_obj': {'dayOfWeek-DayOfWeek', 'price-priceRange'}},
         'JobPosting': {'added_obj': {'image-Photograph'}},
         'Museum': {'added_obj': {'address-Place/name'}},
         'MusicAlbum': {'added_obj': {'image-Photograph','price-priceRange','publisher-Organization'}},
         'MusicRecording': {'added_obj': {'image-Photograph'}},
         'Person': {'added_obj': {'image-Photograph'}},
         'Product': {'added_obj': {'price-priceRange'}},
         'Recipe': {'added_obj': {'image-Photograph'}},
         'SportsEvent': {'added_obj': {'price-priceRange'}},
    }

    topic_bad_cases_po = dict()
    for topic, item in topic_bad_cases.items():
        for case in item['added_obj']:
            if topic not in topic_bad_cases_po:
                topic_bad_cases_po[topic] = dict()
            topic_bad_cases_po[topic][case.split('-')[0]] = case.split('-')[1]

    topic2SPO = {}
    for topic, sp_list in topic2SP.items():

        if topic not in topic2SPO:
            topic2SPO[topic] = list()

        for s, p in sp_list:

            if p not in PO:
                PO[p] = ['?' + p.split('/')[-1]]

            for o in PO[p]:
                if '/description' in o:
                    if s.split('/')[0] != o.split('/')[0]:
                        if s.replace('?', '') not in o:
                            continue
                        else:
                            pass
                if topic in topic_bad_cases_po:
                    if p in topic_bad_cases_po[topic]:
                        if o == topic_bad_cases_po[topic][p]:
                            continue
                if s == o:
                    continue

                topic2SPO[topic].append([s, p, o])

    # pprint.pprint(topic2SPO)
    return topic2SPO

def construct_SPO_with_misalignment(topic2SPO, topic2S):
    # After observing the output from function "detect_misalignment_*"
    # handle missing type here
    # handle added obj/subj in construct SPO func

    t_dict1_add_p = {
        'Boolean': {'Restaurant', 'Hotel', 'Event', 'CreativeWork', 'Book'},
        'CategoryCode': {'JobPosting'},
        'CreativeWork': {'Book', 'Recipe', 'CreativeWork'},
        'Date': {'JobPosting', 'LocalBusiness'},
        'EducationalOccupationalCredential': {'JobPosting'},
        'Energy': {'Recipe'},
        'IdentifierAT': {'JobPosting', 'Person', 'Event', 'LocalBusiness'},
        'ItemList': {'MusicAlbum'},
        'Language': {'Person', 'Event'},
        'MonetaryAmount': {'JobPosting'},
        'MusicArtistAT': {'Movie'},
        'OccupationalExperienceRequirements': {'JobPosting'},
        'Organization': {'Person', 'LocalBusiness'},
        'Person/name': {'LocalBusiness', 'SportsEvent', 'Event', 'MusicAlbum'},
        'Place/name': {'Product', 'CreativeWork'},
        'QuantitativeValue': {'Product', 'Recipe'},
        'Rating': {'Hotel'},
        'Time': {'Recipe', 'Event'},
        'currency': {'JobPosting'},
        'paymentAccepted': {'Product'},
        'workHours': {'JobPosting'}
     }

    topic_o = {
        'Event': {
            'Language': 'inLanguage',
            'Time': '??doorTime'
        },
        'Hotel': {'Rating': '??starRating'},
        'JobPosting': {
            'OccupationalExperienceRequirements':'??experienceRequirements',
            'CategoryCode':'??occupationalCategory',
            'EducationalOccupationalCredential':'??educationRequirements',
            'MonetaryAmount':'??baseSalary/estimatedSalary',
            'currency':'??salaryCurrency',
        },
        'Movie': {'MusicArtistAT': '??musicby'},
        'Person': {'Language': '??knowsLanguage'} ,
        'Product': {
            'DateTime': 'releaseDate',
            'paymentAccepted':'paymentAccepted'
        },
        'Recipe': {
            # 'Time': 'totalTime',
            'CreativeWork': '??recipeInstructions'
        },
        'SportsEvent': {'Person/name': '??referee'}
    }

    for o, topic_list in t_dict1_add_p.items():
        for topic in topic_list:
            if topic not in topic_o:
                continue
            if o in topic_o[topic]:
                p = topic_o[topic][o]
            else:
                p = '??' + o
            topic2SPO[topic].append([topic2S[topic]['subj'], p, o])

    # from missing properties
    for topic in ['CreativeWork', 'LocalBusiness', 'Museum', 'MusicAlbum', 'MusicRecording', 'Person', 'Place', 'Restaurant', 'SportsEvent', 'TVEpisode']:
        topic2SPO[topic].append([topic2S[topic]['subj'], 'description', '?description'])

    for topic in ['Event']:
        topic2SPO[topic].append([topic2S[topic]['subj'], 'addressRegion', 'addressRegion'])

    for topic in ['Hotel']:
        topic2SPO[topic].append([topic2S[topic]['subj'], 'dayOfWeek', '?dayOfWeek'])

    # 'inAlbum'
    for topic in ['MusicAlbum']:
        topic2SPO[topic].append([topic2S[topic]['subj'], 'inAlbum', '?inAlbum'])
        topic2SPO[topic].append([topic2S[topic]['subj'], 'publisher', 'Organization'])

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
    # difference between annotations and tables

    topic2T = list_type_per_topic(file_paths)
    # print ("from annotation: -----------------------" )
    # pprint.pprint(topic2T)
    # func SPO is not good enough
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
                detect_dict[topic]["added_obj"].add(p + '-' + o)


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

    # annotation files (GT)
    node_train_gt_path = get_path(dataset_path, 'CTA_SCH_TRAIN_GT_CSV', 'sotab')
    node_val_gt_path = get_path(dataset_path, 'CTA_SCH_VAL_GT_CSV', 'sotab')
    edge_train_gt_path = get_path(dataset_path, 'CPA_SCH_TRAIN_GT_CSV', 'sotab')
    edge_val_gt_path = get_path(dataset_path, 'CPA_SCH_VAL_GT_CSV', 'sotab')

    file_paths = [node_train_gt_path, node_val_gt_path, edge_train_gt_path, edge_val_gt_path]

    # Step 1: find the main subject candidates based on frequencies
    list_subj_candidate_per_topic(file_paths)

    # Step 2: put the printed dict into a file:topic2type and find out the only relevant subject for each topic based on frequencies manually
    topic2S = {
        'Book': {'subj': 'Book/name'},
        'CreativeWork': {'subj': 'CreativeWork/name'},
        'Event': {'subj': 'Event/name'},
        'Hotel': {'subj': 'Hotel/name'},
        'JobPosting': {'subj': 'JobPosting/name'},
        'LocalBusiness': {'subj': 'LocalBusiness/name'},
        'Movie': {'subj': 'Movie/name'},
        'Museum': {'subj': 'Museum/name'},
        'MusicAlbum': {'subj': 'MusicAlbum/name'},
        'MusicRecording': {'subj': 'MusicRecording/name'},
        'Person': {'subj': 'Person/name'},
        'Place': {'subj': 'Place/name'},
        'Product': {'subj': 'Product/name'},
        'Recipe': {'subj': 'Recipe/name'},
        'Restaurant': {'subj': 'Restaurant/name'},
        'SportsEvent': {'subj': 'SportsEvent/name'},
        'TVEpisode': {'subj': 'TVEpisode/name'}
    }

    # Step 3: Merge main subject with all the properties for each topic
    topic2P = list_property_per_topic(file_paths)
    topic2SP = construct_SP(topic2S, topic2P)
    # pprint.pprint (topic2SP)

    # Step 4 column matching between CTA tables & CPA tables: manual work required
    # property-> type (obejct)
    # (subject/object) type -> property
    type_property_info_by_shared_column(file_paths)

    PO = {
    'actor': {'Person/name'},
    'address': {'PostalAddress'},
    'addressCountry': {'Country'},
    'addressLocality': {'addressLocality'},
    'addressRegion': {'addressRegion'},
    'amenityFeature': {'LocationFeatureSpecification'},
    'author': {'Person/name'},
    'availability': {'ItemAvailability'},
    'availableDeliveryMethod': {'DeliveryMethod'},
    'availableLanguage': {'Language'},
    'awayTeam': {'SportsTeam'},
    'bestRating': {'Number'},
    'birthDate': {'Date'},
    'birthPlace': {'Place/name'},
    'bookFormat': {'BookFormatType'},
    'brand': {'Brand'},
    'byArtist': {'MusicArtistAT'},
    'category': {'category'},
    'checkInTime': {'Time'},
    'checkoutTime': {'Time'},
    'closes': {'Time'},
    'contentRating': {'Rating'},
    'cookTime': {'Duration'},
    'copyrightYear': {'Number'},
    'countryOfOrigin': {'Country'},
    'currenciesAccepted': {'currency'},
    'datePosted': { 'DateTime' },
    'datePublished': {'Date'},
    'dateCreated': {'Date'},
    'dayOfWeek': {'DayOfWeek'},
    'deathDate': {'Date'},
    'description': {'Book/description',
                 'Event/description',
                 'Hotel/description',
                 'JobPosting/description',
                 'Movie/description',
                 'Product/description',
                 'Recipe/description'},
    'director': {'Person/name'},
    'duration': {'Duration'},
    'email': {'email'},
    'employmentType': {'?employmentType' },
    'endDate': {'DateTime'},
    'episodeNumber': { '?episodeNumber' },
    'eventAttendanceMode': {'EventAttendanceModeEnumeration'},
    'eventStatus': {'EventStatusType'},
    'faxNumber': {'faxNumber'},
    'gender': {'GenderType'},
    'genre': { '?genre' },
    'givenName': { '?givenName' },
    'gtin': {'IdentifierAT'},
    'headline': { '?headline' },
    'hiringOrganization': {'Organization'},
    'homeTeam': {'SportsTeam'},
    'image': {'URL'},
    'inAlbum': {'MusicAlbum/name'},
    'inLanguage': {'Language'},
    'isbn': {'IdentifierAT'},
    'itemCondition': {'OfferItemCondition'},
    'jobTitle': { '?jobTitle' },
    'latitude': {'CoordinateAT'},
    'location': {'Place/name'},
    'longitude': {'CoordinateAT'},
    'manufacturer': {'Organization'},
    'material': { '?material' },
    'measurements': {'Distance'},
    'model': {'ProductModel'},
    'nationality': {'Country'},
    'numTracks': { '?numTracks' },
    'numberOfPages': {'Number'},
    'nutrition': {'Mass'},
    'openingHours': {'openingHours'},
    'opens': {'Time'},
    'organizer': {'Organization'},
    'partOfSeries': {'CreativeWorkSeries'},
    'paymentAccepted': {'paymentAccepted'},
    'performer': {'Organization'},
    'postalCode': {'postalCode'},
    'prepTime': {'Duration'},
    'price': {'price'},
    'priceCurrency': {'currency'},
    'priceRange': {'priceRange'},
    'productID': {'IdentifierAT'},
    'productionCompany': {'Organization'},
    'publisher': {'Organization'},
    'ratingValue': {'Number'},
    'recipeIngredient': { '?recipeIngredient' },
    'recipeInstructions': {'ItemList'},
    'releaseDate': {'Date'},
    'review': {'Review'},
    'servesCuisine': { '?servesCuisine' },
    'startDate': {'DateTime'},
    'streetAddress': {'streetAddress'},
    'suitableForDiet': {'RestrictedDiet'},
    'telephone': {'telephone'},
    'text': { '?text' },
    'totalTime': {'Duration'},
    'track': {'MusicRecording/name'},
    'unitCode': {'unitCode'},
    'unitText': {'unitText'},
    'url': {'URL'},
    'validFrom': {'DateTime'},
    'validThrough': {'DateTime'},
    'weight': {'weight'},
    'worstRating': {'Number'}
    }

    # STEP 5 merge
    topic2SPO = construct_SPO(topic2SP, PO)

    # STEP 6 detect if there were 1) missing / 2) new-added / 3) wrongly-added types and properties
    # the information of correct wrongly-added types -> contributes to -> handle special cases with conditions in Step 5 again
    detect_misalignment_type(topic2SPO, file_paths)
    detect_misalignment_property(topic2SPO, file_paths)

    topic2SPO = construct_SPO_with_misalignment(topic2SPO, topic2S)
    # pprint.pprint(topic2SPO)

    SPO = [spo for topic, spo_list in topic2SPO.items() for spo in spo_list]

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
    SPO, _ = table_gt_info()
    for spo in SPO:
        print(spo)
    # create_pg_neo4j(SPO)

if __name__ == "__main__":
    main()