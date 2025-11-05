library(shiny)
library(httr)
library(jsonlite)

# ============================================
# SETUP: Add your Cohere API key here
# ============================================
# Get a free API key at: https://dashboard.cohere.com/api-keys
COHERE_API_KEY <- "YOUR COHERE API HERE"


# Or hardcode it: COHERE_API_KEY <- "your-cohere-api-key-here"

# Parse CSV knowledge base function
parse_csv_knowledge_base <- function(filepath) {
  if (!file.exists(filepath)) {
    stop(paste("CSV file not found:", filepath))
  }
  
  df <- read.csv(filepath, stringsAsFactors = FALSE)
  
  if (!("question" %in% names(df)) || !("answer" %in% names(df))) {
    stop("CSV must contain 'question' and 'answer' columns")
  }
  
  knowledge_base <- list()
  
  for (i in 1:nrow(df)) {
    question <- trimws(as.character(df$question[i]))
    answer <- trimws(as.character(df$answer[i]))
    
    if (question != "" && answer != "" && !is.na(question) && !is.na(answer)) {
      knowledge_base <- append(knowledge_base, list(list(
        question = question,
        answer = answer,
        source = "Training documents"
      )))
    }
  }
  
  if (length(knowledge_base) == 0) {
    stop("No valid Q&A pairs found in CSV")
  }
  
  return(knowledge_base)
}

# Get Cohere embeddings in BATCH (up to 96 at a time)
get_cohere_embeddings_batch <- function(texts, api_key = COHERE_API_KEY, batch_size = 96) {
  if (api_key == "" || is.null(api_key)) {
    return(NULL)
  }
  
  all_embeddings <- list()
  num_batches <- ceiling(length(texts) / batch_size)
  
  cat(sprintf("📦 Processing %d texts in %d batch(es)...\n", length(texts), num_batches))
  
  for (batch_num in 1:num_batches) {
    start_idx <- (batch_num - 1) * batch_size + 1
    end_idx <- min(batch_num * batch_size, length(texts))
    batch_texts <- texts[start_idx:end_idx]
    
    cat(sprintf("  Batch %d/%d (questions %d-%d)...", batch_num, num_batches, start_idx, end_idx))
    
    tryCatch({
      response <- POST(
        "https://api.cohere.com/v1/embed",
        add_headers(
          "Authorization" = paste("Bearer", api_key),
          "Content-Type" = "application/json"
        ),
        body = toJSON(list(
          model = "embed-english-light-v3.0",  # Fast, free tier friendly
          texts = batch_texts,
          input_type = "search_document"
        ), auto_unbox = TRUE),
        encode = "json",
        timeout(60)
      )
      
      if (status_code(response) == 200) {
        result <- content(response, as = "parsed")
        
        # Extract embeddings
        for (embedding in result$embeddings) {
          all_embeddings[[length(all_embeddings) + 1]] <- unlist(embedding)
        }
        
        cat(" ✅\n")
        
        # Small delay between batches
        if (batch_num < num_batches) {
          Sys.sleep(1)
        }
        
      } else if (status_code(response) == 429) {
        cat(" ⏳ Rate limit. Waiting 10s...\n")
        Sys.sleep(10)
        batch_num <- batch_num - 1  # Retry this batch
      } else {
        cat(sprintf(" ❌ Error %d\n", status_code(response)))
        # Add NULL for failed embeddings
        for (i in start_idx:end_idx) {
          all_embeddings[[length(all_embeddings) + 1]] <- NULL
        }
      }
    }, error = function(e) {
      cat(sprintf(" ❌ %s\n", e$message))
      for (i in start_idx:end_idx) {
        all_embeddings[[length(all_embeddings) + 1]] <- NULL
      }
    })
  }
  
  return(all_embeddings)
}

# Single embedding for user queries
get_cohere_embedding_single <- function(text, api_key = COHERE_API_KEY) {
  if (api_key == "" || is.null(api_key)) {
    return(NULL)
  }
  
  tryCatch({
    response <- POST(
      "https://api.cohere.com/v1/embed",
      add_headers(
        "Authorization" = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      ),
      body = toJSON(list(
        model = "embed-english-light-v3.0",
        texts = list(text),
        input_type = "search_query"
      ), auto_unbox = TRUE),
      encode = "json",
      timeout(30)
    )
    
    if (status_code(response) == 200) {
      result <- content(response, as = "parsed")
      return(unlist(result$embeddings[[1]]))
    } else {
      return(NULL)
    }
  }, error = function(e) {
    return(NULL)
  })
}

# Cosine similarity
cosine_similarity <- function(vec1, vec2) {
  dot_product <- sum(vec1 * vec2)
  magnitude1 <- sqrt(sum(vec1^2))
  magnitude2 <- sqrt(sum(vec2^2))
  return(dot_product / (magnitude1 * magnitude2))
}

# Load knowledge base
cat("🔄 Loading knowledge base from CSV...\n")
kb <- parse_csv_knowledge_base("knowledge_base_ai.csv")
cat("✅ Loaded", length(kb), "Q&A pairs\n")

# Try to load cached embeddings
embeddings_cache_file <- "embeddings_cache.rds"
question_embeddings <- list()
use_ai <- FALSE

if (file.exists(embeddings_cache_file)) {
  cat("📂 Loading cached embeddings...\n")
  tryCatch({
    cached_data <- readRDS(embeddings_cache_file)
    
    if (length(cached_data$embeddings) == length(kb)) {
      questions_match <- TRUE
      for (i in 1:min(10, length(kb))) {
        if (kb[[i]]$question != cached_data$questions[i]) {
          questions_match <- FALSE
          break
        }
      }
      
      if (questions_match) {
        question_embeddings <- cached_data$embeddings
        cat("✅ Loaded", length(question_embeddings), "cached embeddings!\n")
        use_ai <- TRUE
      }
    }
  }, error = function(e) {
    cat("⚠️  Could not load cache\n")
  })
}

# Generate embeddings if not cached
if (!use_ai && COHERE_API_KEY != "" && !is.null(COHERE_API_KEY)) {
  cat("🔄 Generating Cohere embeddings...\n")
  cat("💡 Cohere allows 96 texts per batch - much faster!\n")
  
  all_questions <- sapply(kb, function(x) x$question)
  
  question_embeddings <- get_cohere_embeddings_batch(all_questions, batch_size = 96)
  
  if (!is.null(question_embeddings) && length(question_embeddings) > 0) {
    success_count <- sum(sapply(question_embeddings, function(x) !is.null(x)))
    cat(sprintf("✅ Generated %d/%d embeddings!\n", success_count, length(kb)))
    
    # Save cache
    if (success_count > 0) {
      tryCatch({
        saveRDS(list(
          embeddings = question_embeddings,
          questions = all_questions,
          timestamp = Sys.time()
        ), embeddings_cache_file)
        cat("💾 Embeddings cached!\n")
      }, error = function(e) {
        cat("⚠️  Could not save cache\n")
      })
    }
    
    use_ai <- (success_count > 0)
  }
}

if (!use_ai) {
  cat("⚠️  Using keyword matching mode\n")
}

# AI-powered answer matching
find_best_answer_ai <- function(user_question, min_confidence = 30) {
  if (trimws(user_question) == "") {
    return("Please ask me a question about city services!")
  }
  
  user_embedding <- get_cohere_embedding_single(user_question)
  
  if (is.null(user_embedding)) {
    return("Sorry, I'm having trouble processing your question. Please try again or call 311 at 617-666-3311.")
  }
  
  similarities <- numeric(length(kb))
  for (i in 1:length(kb)) {
    if (!is.null(question_embeddings[[i]])) {
      similarities[i] <- cosine_similarity(user_embedding, question_embeddings[[i]])
    }
  }
  
  confidences <- similarities * 100
  good_matches <- which(confidences >= min_confidence)
  
  if (length(good_matches) == 0) {
    return("I don't have specific information about that. Call 311 (617-666-3311) or visit somervillema.gov for assistance.")
  }
  
  good_matches <- good_matches[order(confidences[good_matches], decreasing = TRUE)]
  
  if (length(good_matches) > 5) {
    good_matches <- good_matches[1:5]
  }
  
  response_parts <- c()
  
  for (i in seq_along(good_matches)) {
    match_idx <- good_matches[i]
    match_item <- kb[[match_idx]]
    match_confidence <- confidences[match_idx]
    
    result_header <- sprintf("📋 **Result %d** (🎯 %.0f%% match)", i, match_confidence)
    result_question <- paste("**Q:**", match_item$question)
    result_answer <- paste("**A:**", match_item$answer)
    
    response_parts <- c(response_parts, result_header, result_question, result_answer, "")
  }
  
  summary <- sprintf("Found %d result%s above %.0f%% confidence:", 
                     length(good_matches), 
                     ifelse(length(good_matches) == 1, "", "s"), 
                     min_confidence)
  
  final_response <- paste(c(summary, "", response_parts), collapse = "\n")
  
  return(final_response)
}

# Keyword matching fallback
find_best_answer_keywords <- function(user_question, min_confidence = 30) {
  if (trimws(user_question) == "") {
    return("Please ask me a question about city services!")
  }
  
  user_lower <- tolower(user_question)
  scores <- numeric(length(kb))
  
  stop_words <- c("i", "me", "my", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "can", "do", "does", "is", "are", "was", "were", "be", "been", "have", "has", "had", "will", "would", "could", "should")
  
  user_words <- strsplit(gsub("[^a-zA-Z0-9 ]", "", user_lower), " ")[[1]]
  user_words <- user_words[nchar(user_words) > 2]
  user_words <- user_words[!user_words %in% stop_words]
  
  for (i in 1:length(kb)) {
    item <- kb[[i]]
    question_lower <- tolower(item$question)
    answer_lower <- tolower(item$answer)
    
    if (length(user_words) == 0) {
      scores[i] <- 0
      next
    }
    
    question_matches <- 0
    answer_matches <- 0
    
    for (word in unique(user_words)) {
      if (grepl(word, question_lower)) question_matches <- question_matches + 1
      if (grepl(word, answer_lower)) answer_matches <- answer_matches + 1
    }
    
    score <- (question_matches * 3) + (answer_matches * 2)
    word_match_ratio <- (question_matches + answer_matches) / length(user_words)
    if (word_match_ratio < 0.3) score <- score * 0.5
    
    scores[i] <- score
  }
  
  max_possible_score <- length(unique(user_words)) * 3
  confidences <- pmin(100, (scores / max_possible_score) * 100)
  confidences[is.nan(confidences)] <- 0
  
  good_matches <- which(confidences >= min_confidence)
  
  if (length(good_matches) == 0) {
    return("I don't have specific information about that. Call 311 (617-666-3311) or visit somervillema.gov for assistance.")
  }
  
  good_matches <- good_matches[order(confidences[good_matches], decreasing = TRUE)]
  if (length(good_matches) > 5) good_matches <- good_matches[1:5]
  
  response_parts <- c()
  for (i in seq_along(good_matches)) {
    match_idx <- good_matches[i]
    match_item <- kb[[match_idx]]
    match_confidence <- confidences[match_idx]
    
    result_header <- sprintf("📋 **Result %d** (🎯 %.0f%% match)", i, match_confidence)
    result_question <- paste("**Q:**", match_item$question)
    result_answer <- paste("**A:**", match_item$answer)
    
    response_parts <- c(response_parts, result_header, result_question, result_answer, "")
  }
  
  summary <- sprintf("Found %d result%s above %.0f%% confidence:", 
                     length(good_matches), 
                     ifelse(length(good_matches) == 1, "", "s"), 
                     min_confidence)
  
  final_response <- paste(c(summary, "", response_parts), collapse = "\n")
  return(final_response)
}

# Main answer function
find_best_answer <- function(user_question, min_confidence = 30) {
  if (use_ai) {
    return(find_best_answer_ai(user_question, min_confidence))
  } else {
    return(find_best_answer_keywords(user_question, min_confidence))
  }
}

# UI
ui <- fluidPage(
  titlePanel("🏛️ Somerville 311 Assistant"),
  
  fluidRow(
    column(4,
           wellPanel(
             h4("📚 Knowledge Base"),
             p(strong(paste("📊", length(kb), "topics available"))),
             p(if(use_ai) "🤖 Using Cohere AI semantic matching" else "🔤 Using keyword matching"),
             
             hr(),
             h5("🎛️ Search Settings:"),
             sliderInput("min_confidence", 
                         "Minimum Confidence %:", 
                         min = 10, max = 80, value = 30, step = 5),
             p(style = "font-size: 11px; color: #666;", 
               "Lower values show more results, higher values show only the best matches."),
             
             hr(),
             h5("💡 Try asking about:"),
             div(
               actionButton("ex1", "Parking permits", class = "btn-sm btn-outline-primary", style = "margin: 2px;"),
               actionButton("ex2", "Trash pickup", class = "btn-sm btn-outline-primary", style = "margin: 2px;"),
               actionButton("ex3", "Water bills", class = "btn-sm btn-outline-primary", style = "margin: 2px;"),
               actionButton("ex4", "Voter registration", class = "btn-sm btn-outline-primary", style = "margin: 2px;")
             )
           )
    ),
    
    column(8,
           wellPanel(
             h4("💬 Chat"),
             div(
               id = "chat-container",
               style = "height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; background: #f9f9f9;",
               uiOutput("chat_display")
             )
           ),
           
           fluidRow(
             column(12,
                    textAreaInput("user_input", 
                                  label = NULL,
                                  placeholder = "Ask about city services...",
                                  width = "100%",
                                  rows = 2,
                                  resize = "vertical")
             )
           ),
           fluidRow(
             column(12,
                    actionButton("send_btn", "Send 📤", 
                                 class = "btn-primary btn-lg",
                                 style = "width: 100%;")
             )
           ),
           
           br(),
           p(class = "text-muted", style = "font-size: 12px;", 
             if(use_ai) "🤖 Cohere AI semantic search • Returns exact answers from knowledge base" else "🔤 Keyword matching • Returns exact answers from knowledge base")
    )
  ),
  
  tags$head(
    tags$style(HTML("
      #chat-container {
        overflow-y: auto !important;
        overflow-x: hidden !important;
      }
      .chat-message {
        margin: 10px 5px;
        padding: 12px 15px;
        border-radius: 15px;
        max-width: 85%;
        word-break: break-word;
        overflow-wrap: anywhere;
        hyphens: auto;
      }
      .user-message {
        background: #007bff;
        color: white;
        margin-left: auto;
        text-align: left;
        float: right;
        clear: both;
      }
      .bot-message {
        background: white;
        border: 1px solid #dee2e6;
        white-space: pre-wrap;
        word-break: break-word;
        float: left;
        clear: both;
      }
      .clearfix {
        clear: both;
      }
    "))
  )
)

# Server
server <- function(input, output, session) {
  
  chat_history <- reactiveVal(list())
  
  add_message <- function(text, is_user = TRUE) {
    current <- chat_history()
    new_msg <- list(
      text = as.character(text),
      is_user = is_user,
      id = paste0("msg_", length(current) + 1),
      timestamp = Sys.time()
    )
    
    chat_history(append(current, list(new_msg)))
  }
  
  output$chat_display <- renderUI({
    messages <- chat_history()
    
    if (length(messages) == 0) {
      return(div(
        class = "alert alert-info",
        if(use_ai) {
          "👋 Hello! I'm using Cohere AI to intelligently match your questions. Ask me anything!"
        } else {
          "👋 Hello! I'm here to help with Somerville city services. Ask me anything!"
        }
      ))
    }
    
    message_divs <- lapply(messages, function(msg) {
      class_name <- if (msg$is_user) "chat-message user-message" else "chat-message bot-message"
      div(
        class = class_name,
        HTML(gsub("\n", "<br>", msg$text))
      )
    })
    
    all_divs <- lapply(message_divs, function(d) {
      tagList(d, div(class = "clearfix"))
    })
    
    return(tagList(all_divs))
  })
  
  process_message <- function() {
    user_text <- trimws(input$user_input)
    if (user_text == "") return()
    
    add_message(user_text, TRUE)
    
    bot_response <- find_best_answer(user_text, input$min_confidence)
    add_message(bot_response, FALSE)
    
    updateTextAreaInput(session, "user_input", value = "")
    session$sendCustomMessage("scrollToBottom", list())
  }
  
  observeEvent(input$send_btn, {
    process_message()
  })
  
  observeEvent(input$ex1, {
    updateTextAreaInput(session, "user_input", value = "How do I get a parking permit?")
  })
  
  observeEvent(input$ex2, {
    updateTextAreaInput(session, "user_input", value = "When is my trash picked up?")
  })
  
  observeEvent(input$ex3, {
    updateTextAreaInput(session, "user_input", value = "How do I pay my water bill?")
  })
  
  observeEvent(input$ex4, {
    updateTextAreaInput(session, "user_input", value = "How do I register to vote?")
  })
}

ui <- tagList(
  ui,
  tags$script(HTML("
    Shiny.addCustomMessageHandler('scrollToBottom', function(message) {
      var container = document.getElementById('chat-container');
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    });
    
    $(document).on('keydown', '#user_input', function(e) {
      if (e.which == 13 && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        $('#send_btn').click();
      }
    });
    
    $(document).on('shiny:value', function(event) {
      if (event.name === 'chat_display') {
        setTimeout(function() {
          var container = document.getElementById('chat-container');
          if (container) {
            container.scrollTop = container.scrollHeight;
          }
        }, 100);
      }
    });
  "))
)

cat("🚀 Starting chatbot...\n")
if (use_ai) {
  cat("✅ Cohere AI mode enabled\n")
} else {
  cat("⚠️  Keyword matching mode\n")
  cat("💡 To enable Cohere AI: Get free API key from https://dashboard.cohere.com/api-keys\n")
}


shinyApp(ui = ui, server = server)
