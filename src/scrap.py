import snscrape.modules.twitter as twitterScrapper
import pandas as pd

def scrap_user(user):
    """ Devuelve un máximo de 101 tweets por usuario. """
    # Datos elegidos.
    ATTR_LIST = ["date", "content", "username", "description", "display_name","verified", "followersCount", "friendsCount",
                "tweetsCount", "listedCount", "mediaCount", "language", "replyCount", "retweetCount", "likeCount",
                "quoteCount", "sourceUrl", "sourceLabel"]

    datos = []
    sc = twitterScrapper.TwitterUserScraper(user)
    if sc is None:
        raise Exception("Error. This user does not exist or it's a private account.")
    # Si el usuario no existe se lanza una excepción.
    try:
        sc.entity
    except KeyError:
        raise Exception("Error. This user does not exist or it's a private account.")
    if sc.entity is None:
        raise Exception("Error. This user does not exist or it's a private account.")
    # Máximo de 101 tweets.
    limite = 101
    for i, tw in enumerate(sc.get_items()):
        if i > limite:
            break
        # Saltamos los retweets.
        if tw.retweetedTweet is not None:
            limite+=1
            continue
        contenido = tw.renderedContent.replace(",", "")
        contenido = contenido.replace("\n", " ")
        desc = tw.user.rawDescription.replace(",", "")
        desc = desc.replace("\n", " ")
        datos.append({ATTR_LIST[0]: tw.date, ATTR_LIST[1]: contenido, ATTR_LIST[2]: tw.user.username, 
            ATTR_LIST[3]: desc, ATTR_LIST[4]: tw.user.displayname, ATTR_LIST[5]: tw.user.verified, 
            ATTR_LIST[6]: tw.user.followersCount, ATTR_LIST[7]: tw.user.friendsCount, ATTR_LIST[8]: tw.user.statusesCount, 
            ATTR_LIST[9]: tw.user.listedCount, ATTR_LIST[10]: tw.user.mediaCount, ATTR_LIST[11]: tw.lang,
            ATTR_LIST[12]: tw.replyCount, ATTR_LIST[13]: tw.retweetCount, ATTR_LIST[14]: tw.likeCount,
            ATTR_LIST[15] : tw.quoteCount, ATTR_LIST[16]: tw.sourceUrl, ATTR_LIST[17]: tw.sourceLabel})

    output = pd.DataFrame(datos, columns=ATTR_LIST)
    return output
    # output.to_csv("output.csv")
