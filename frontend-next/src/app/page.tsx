'use client'

import { useState, useRef, useEffect } from 'react'
import TypingIndicator from '@/components/TypingIndicator'

interface Message {
  id: number
  text: string
  sender: 'user' | 'yori'
  timestamp: Date
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isTyping])

  const sendMessage = async () => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const messageToSend = inputValue
    setInputValue('')

    // Show typing indicator after 800ms
    setTimeout(() => {
      setIsTyping(true)
    }, 800)

    try {
      const response = await fetch('https://27u5e78ip5v3e0-8000.proxy.runpod.net/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageToSend,
          user_id: 'test_user'
        })
      })

      const data = await response.json()
      
      setIsTyping(false)
      
      const yoriMessage: Message = {
        id: Date.now() + 1,
        text: data.message || data.response || 'Sorry, I couldn\'t process that.',
        sender: 'yori',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, yoriMessage])
    } catch (error) {
      setIsTyping(false)
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: 'Sorry, I\'m having trouble connecting right now.',
        sender: 'yori',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage()
  }

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Header */}
      <div className="bg-gray-50 border-b border-gray-200 p-4 text-center flex-shrink-0">
        <h1 className="text-lg font-semibold text-gray-800">Yori</h1>
        <div className="w-3 h-3 bg-green-500 rounded-full mx-auto mt-1"></div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-20">
            <p className="text-lg">Start a conversation with Yori</p>
            <p className="text-sm mt-2">Type a message below to get started</p>
          </div>
        )}
        
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-2xl ${
                message.sender === 'user'
                  ? 'bg-blue-500 text-white rounded-br-md'
                  : 'bg-gray-200 text-gray-800 rounded-bl-md'
              }`}
            >
              <p className="text-sm leading-relaxed">{message.text}</p>
            </div>
          </div>
        ))}
        
        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-gray-200 px-4 py-2 rounded-2xl rounded-bl-md">
              <TypingIndicator />
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4 flex-shrink-0">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="iMessage"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            disabled={isTyping}
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isTyping}
            className="px-6 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}