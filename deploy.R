library(rsconnect)

rsconnect::setAccountInfo(
  name='andrewnee29', 
  token='D8F2450958E2F0AE729CA15AB3A92166', 
  secret='IJAKIui3hybE1yUSKAqpNxrcXsguGGSQz6vZuM/T'
)

rsconnect::deployApp(
  'C:\\Users\\anee\\Repos\\311 chatbot exploration',
  appFiles = c('app.R', 'knowledge_base_ai.csv', 'embeddings_cache.rds')
)