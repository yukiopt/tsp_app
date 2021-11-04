import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pulp
import folium
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import random
from pyproj import  Geod

@st.cache
def load_data():

    # 全国の地方自治体データ
    localgov_df = pd.read_csv('localgov.csv', usecols=['pref', 'cid', 'city', 'lat', 'lng'])
    # 政令指定都市は除く
    seirei = ('札幌市', '仙台市', 'さいたま市', '千葉市', '横浜市', '川崎市', '相模原市', '新潟市', '静岡市', '浜松市',
            '名古屋市', '京都市', '大阪市', '堺市', '神戸市', '岡山市', '広島市', '北九州市', '福岡市', '熊本市')
    localgov_df = localgov_df[~localgov_df.city.isin(seirei)]

    # 大圏距離を計算してデータフレームにする
    df = localgov_df.loc[:, ['cid', 'lat', 'lng']]
    df['key'] = 1
    df = pd.merge(df, df, on='key', how='outer',suffixes=['1', '2']).drop('key', axis=1)
    geod = Geod(ellps='WGS84')
    def get_distance(lat1, lng1, lat2, lng2):
        _, _, dist = geod.inv(lng1, lat1, lng2, lat2)
        return dist
    df['distance'] = get_distance(df.lat1.tolist(), df.lng1.tolist(), df.lat2.tolist(), df.lng2.tolist())
    df.distance = df.distance / 1000
    df = df.loc[:, ['cid1', 'cid2', 'distance']]

    return localgov_df, df

@st.cache
def tsp_subtour_elimination(localgov_df, distance_df):
    # 辞書の作成
    localgov_dic = localgov_df.set_index('cid').to_dict(orient='index')
    distance_dic = distance_df.set_index(['cid1', 'cid2']).to_dict(orient='index')
    nodes = list(localgov_dic.keys())
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edges = [(i, j) for i in nodes for j in nodes if i < j]

    # 最適化モデルの定義
    model = pulp.LpProblem('', sense=pulp.LpMinimize)

    # 決定変数x。枝(i, j)を使うか否か。
    x = {(i, j):pulp.LpVariable(f'edge({i}, {j})', cat='Binary') for (i, j) in edges}

    # 制約条件。各都市を必ず1度通る。
    for i in nodes:
        avlbl_edges = [(j, i) for j in nodes if (j, i) in edges] + [(i, j) for j in nodes if (i, j) in edges]
        model += pulp.lpSum( x[edge] for edge in avlbl_edges ) == 2

    # 目的関数。総距離。
    model += pulp.lpSum( distance_dic[i, j]['distance'] * x[i, j] for (i, j) in edges )

    # この条件で解く
    model.solve()

    # 使われる枝
    used_edges = [(i, j) for (i, j) in edges if x[i, j].value() > 0.5]
    G.add_edges_from(used_edges)

    # 部分巡回路除去
    CC = list(nx.connected_components(G))
    while len(CC) > 1:
        for S in CC:
            model += pulp.lpSum( x[i, j] for (i, j) in edges if i in S and j in S ) <= len(S) - 1
        status = model.solve()

        G.remove_edges_from(used_edges)
        used_edges = [(i, j) for (i, j) in edges if x[i, j].value() > 0.5]
        G.add_edges_from(used_edges)
        CC = list(nx.connected_components(G))

    # 最適な巡回路
    tour = [(i, j) for (i, j) in x if x[i, j].value() >= 0.5]

    return localgov_dic, pulp.LpStatus[status], tour, model.objective.value()

@st.cache
def tsp_nearest_neighbor(localgov_df, distance_df):
    # 辞書の作成
    localgov_dic = localgov_df.set_index('cid').to_dict(orient='index')
    distance_dic = distance_df.set_index(['cid1', 'cid2']).to_dict(orient='index')
    # 現在の都市から最も近い都市を見つける関数
    def find_nearest_city(current_city, cities):
        return min(cities, key=lambda x:distance_dic[current_city, x]['distance'])
    # Nearest Neighbor法
    cities = list(localgov_dic.keys())
    start_city = random.choice(cities)
    tour = [start_city]
    tours = [tour.copy()]
    unvisited_cities = set(cities) - {start_city}
    while len(unvisited_cities) > 0:
        next_city = find_nearest_city(tour[-1], unvisited_cities)
        tour.append(next_city)
        tours.append(tour.copy())
        unvisited_cities.remove(next_city)
    tour.append(start_city)
    tours.append(tour.copy())
    # 総距離
    TourLength = 0
    for i in range(len(tour)-1):
        TourLength += distance_dic[tour[i], tour[i+1]]['distance']
 
    return localgov_dic, tour, TourLength, tours

@st.cache
def tsp_2_opt(localgov_df, distance_df):
    # 辞書の作成
    localgov_dic = localgov_df.set_index('cid').to_dict(orient='index')
    distance_dic = distance_df.set_index(['cid1', 'cid2']).to_dict(orient='index')
    cities = list(localgov_dic.keys())
    tour = random.sample(cities, len(cities))
    tour.append(tour[0])
    tours = [tour.copy()]
    n = len(tour)
    # 2-opt法
    while True:
        count = 0
        for i in range(n-2):
            i1 = i + 1
            for j in range(i+2, n):
                if j == n - 1:
                    j1 = 0
                else:
                    j1 = j + 1
                if i != 0 or j1 != 0:
                    l1 = distance_dic[tour[i], tour[i1]]['distance']
                    l2 = distance_dic[tour[j], tour[j1]]['distance']
                    l3 = distance_dic[tour[i], tour[j]]['distance']
                    l4 = distance_dic[tour[i1], tour[j1]]['distance']
                    if l1 + l2 > l3 + l4:
                        new_path = tour[i1:j+1]
                        tour[i1:j+1] = new_path[::-1]
                        tours.append(tour.copy())
                        count += 1
        if count==0:
            break
    # 総距離
    TourLength = 0
    for i in range(len(tour)-1):
        TourLength += distance_dic[tour[i], tour[i+1]]['distance']
 
    return localgov_dic, tour, TourLength, tours

# データを読み込む
localgov_df, distance_df = load_data()

st.title('巡回セールスマン問題で遊ぼう！')

st.markdown('### ０.はじめに')

"""
これは、数理最適化の典型問題の一つ、[巡回セールスマン問題](https://ja.wikipedia.org/wiki/%E5%B7%A1%E5%9B%9E%E3%82%BB%E3%83%BC%E3%83%AB%E3%82%B9%E3%83%9E%E3%83%B3%E5%95%8F%E9%A1%8C)で遊ぶアプリです。  
全国の地方自治体を最短距離で回るルートの可視化を是非お楽しみください！  
　　
※都道府県を複数選択できますが、地方自治体の数が多すぎると最適化計算やグラフ作成に時間がかかりすぎることがあるのでご注意ください。  
※都度上から処理されるので、チェックの外し忘れ等にご注意ください！  
"""
if st.checkbox('出典・参考文献など', value=False):
    """
    1. 地方自治体の座標等の情報は以下から取得  
    https://github.com/code4fukui/localgovjp  
      
    2. 厳密解法は以下の書籍を参考に実装  
    [Pythonによる数理最適化入門](https://www.asakura.co.jp/detail.php?book_code=12895)  
      
    3. Nearest Neighbor法は以下の記事を参考に実装  
    [巡回セールスマン問題実践集②~最近傍探索法編~](https://qiita.com/spwimdar/items/346b2ce018bc21857ea15)  
      
    4. 2-opt法は以下の記事を参考に実装  
    http://www.nct9.ne.jp/m_hiroi/light/pyalgo64.html
      
    5. こちらの動画をご参照いただけると、巡回セールスマン問題の歴史や高速に解く手法を知る足がかりになります  
    [巡回セールスマン問題の最先端](https://qiita.com/spwimdar/items/346b2ce018bc21857ea15)

    6. 巡回セールスマン問題の解法についてはこちらの動画をご参照ください  
    [ipad版CONCORDEを用いて巡回セールスマン問題のアルゴリズムを解説](https://youtu.be/yYhnvbGVT8g)
    """

st.markdown('### １. 地図で確認')

# 座標を地図で確認
if st.checkbox('地方自治体の位置を地図で確認', value=False):
    st.markdown('#### 地方自治体マップ')
    # 地図
    color = st.color_picker('マーカーの色を選択', value='#F34F4F')
    localgovs_map = folium.Map(location=[37.5, 138.0], tiles='cartodbpositron', zoom_start=5)
    for i, row in localgov_df.iterrows():
        folium.Circle(
            location=[row.lat, row.lng]
            , popup=row.city
            , radius=1000
            , fill=True
            , fill_color=color
            , color=color
        ).add_to(localgovs_map)
    # 表示
    st.components.v1.html(folium.Figure().add_child(localgovs_map).render(), height=500)

st.markdown('### ２. ルートを最適化する')

# 都道府県の選択
with st.form(key='prefs'):
    selected_prefs = st.multiselect('都道府県を選択', localgov_df.pref.unique(), default='神奈川県')
    submit_button = st.form_submit_button('都道府県を確定')

# 地方自治体データ, 距離データを選択された都道府県に絞る
localgov_df = localgov_df[localgov_df.pref.isin(selected_prefs)]
distance_df = distance_df[(distance_df.cid1.isin(localgov_df.cid.unique()) & distance_df.cid2.isin(localgov_df.cid.unique()))]
n_city = len(localgov_df)
st.write('地方自治体の数：', n_city)

# 手法の選択
method = st.radio('ルート最適化の手法', options=('厳密解法（部分巡回路除去定式化）', 'ヒューリスティクス（Nearest Neighbor法）', 'ヒューリスティクス（2-opt法）'))   

# 実行していいか判定
if method == '厳密解法(部分巡回路除去定式化)' and n_city > 200:
    """
    計算に時間がかかりすぎる場合があります。  
    地方自治体の数が200以内になるように都道府県を選び直してください。
    """
    start = False
else:
    start = st.checkbox('計算開始', value=False)
    

if start:
    if method == '厳密解法（部分巡回路除去定式化）':
        st.write('Now solving...')
        localgov_dic, status, routes, TourLength = tsp_subtour_elimination(localgov_df, distance_df)
        st.write('Solved!')

        st.markdown('### ３. 結果を確認する')

        if st.checkbox('結果を表示', value=False):
            st.write('総距離：', round(TourLength, 1), 'km')
            tour_map = folium.Map(location=[localgov_df.lat.mean(), localgov_df.lng.mean()], tiles='cartodbpositron', zoom_start=9)
            # 巡回路
            for (city_i, city_j) in routes:
                both_ends = [
                    [localgov_dic[city_i]['lat'], localgov_dic[city_i]['lng']]
                    , [localgov_dic[city_j]['lat'], localgov_dic[city_j]['lng']]
                ]
                folium.vector_layers.PolyLine(
                    locations=both_ends
                ).add_to(tour_map)
            # 各地方自治体の位置
            for city in localgov_dic:
                folium.Circle(
                    location=[localgov_dic[city]['lat'], localgov_dic[city]['lng']]
                    , popup=localgov_dic[city]['city']
                    , radius=800
                    , fill=True
                    , fill_color='#706A73'
                    , color='#706A73'
                ).add_to(tour_map)
            # 表示
            st.components.v1.html(folium.Figure().add_child(tour_map).render(), height=500)

    else:
        if method == 'ヒューリスティクス（Nearest Neighbor法）':
            localgov_dic, tour, TourLength, tours = tsp_nearest_neighbor(localgov_df, distance_df)
        else :
            localgov_dic, tour, TourLength, tours = tsp_2_opt(localgov_df, distance_df)
        st.write('Solved!')
        # アニメーション
        if st.checkbox('解く過程を表示', value=False):
            """
            地方自治体の数が多いと、アニメーション表示までに時間がかかったり、  
            最後まで表示されない可能性があります。
            """
            fig = plt.figure()
            def plot(i):
                if i != 0:
                    plt.cla()
                plt.xlim([localgov_df.lng.min() - 0.05, localgov_df.lng.max() + 0.05])
                plt.ylim([localgov_df.lat.min() - 0.05, localgov_df.lat.max() + 0.05])
                plt.xlabel('longitude')
                plt.ylabel('latitude')
                plt.title('Tour')
                lng_list = [(localgov_dic[city]['lng']) for city in tours[i]]
                lat_list = [(localgov_dic[city]['lat']) for city in tours[i]]
                # 都市
                plt.scatter(x=localgov_df.lng, y=localgov_df.lat, color='dimgray')
                # 巡回路
                plt.plot(
                    lng_list,
                    lat_list,
                    color='royalblue'
                )

            ani = animation.FuncAnimation(fig, plot, frames=len(tours), interval=100)
            components.html(ani.to_jshtml(), height=600)
        
        st.markdown('### ３. 結果を確認する')
        if st.checkbox('結果を表示', value=False):
            st.write('総距離：', round(TourLength, 1), 'km')
            tour_map = folium.Map(location=[localgov_df.lat.mean(), localgov_df.lng.mean()], tiles='cartodbpositron', zoom_start=9)
            # 巡回路
            coordinates = [[localgov_dic[city]['lat'], localgov_dic[city]['lng']] for city in tour]
            folium.vector_layers.PolyLine(
                    locations=coordinates
                ).add_to(tour_map)
            # 各地方自治体の位置
            for city in localgov_dic:
                folium.Circle(
                    location=[localgov_dic[city]['lat'], localgov_dic[city]['lng']]
                    , popup=localgov_dic[city]['city']
                    , radius=800
                    , fill=True
                    , fill_color='#706A73'
                    , color='#706A73'
                ).add_to(tour_map)
            # 表示
            st.components.v1.html(folium.Figure().add_child(tour_map).render(), height=500)
