'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import TypingIndicator from '@/components/TypingIndicator'

interface Message {
  id: number
  text: string
  sender: 'user' | 'yori'
  timestamp: Date
}

const DEFAULT_API_BASE = "https://jji00yxx1xwxn1-8000.proxy.runpod.net/";
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_BASE).replace(/\/$/, '')
const CHAT_ENDPOINT = `${API_BASE_URL}/chat`
const TYPING_DELAY_MS = 800

const buildMessage = (text: string, sender: Message['sender']): Message => ({
  id: Date.now() + Math.random(),
  text,
  sender,
  timestamp: new Date()
})

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const typingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, isTyping, scrollToBottom])

  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current)
      }
    }
  }, [])

  const handleSend = useCallback(async () => {
    const trimmed = inputValue.trim()
    if (!trimmed) return

    setMessages(prev => [...prev, buildMessage(trimmed, 'user')])
    setInputValue('')

    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current)
    }

    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(true)
    }, TYPING_DELAY_MS)

    try {
      const response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: trimmed,
          user_id: 'test_user'
        })
      })

      if (!response.ok) {
        throw new Error(`Chat request failed with status ${response.status}`)
      }

      const data = await response.json()
      const reply = data.response ?? data.message ?? "Sorry, I couldn't process that."

      setMessages(prev => [...prev, buildMessage(reply, 'yori')])
    } catch (error) {
      console.error('Failed to send chat message', error)
      setMessages(prev => [
        ...prev,
        buildMessage("Sorry, I'm having trouble connecting right now.", 'yori')
      ])
    } finally {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current)
        typingTimeoutRef.current = null
      }
      setIsTyping(false)
    }
  }, [inputValue])

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    handleSend()
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
            onKeyDown={handleKeyDown}
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
