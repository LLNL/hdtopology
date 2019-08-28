#ifndef MEMBUF_HPP
# define MEMBUF_HPP

#include <cassert>
#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <streambuf>
#include <cstdio>



class autoResizeMembuf : public std::streambuf {
public:
  autoResizeMembuf(bool isOutput = true, unsigned int len = 10240):offset(0), allocationIncrement(20000)
  {
    // fprintf(stderr, "!!! init autoResizeMembuf !!!\n");
    // allocationIncrement = 2000;
    buffer.resize(len);
    this->begin_ = (const char*)&buffer[0];
    this->end_ = (const char*)&buffer[0]+buffer.size();
    this->current_ = (const char*)&buffer[0];

    if(isOutput)
      this->setp( (char*)begin_, (char*)end_ );
  }

  size_t outputCount(){
    //new range + old range
    return this->pptr() - this->pbase() + offset;
  }

  void setBuffer(std::vector<char> &inputBuffer){
    buffer = inputBuffer;
    this->begin_ = (const char*)&buffer[0];
    this->end_ = (const char*)&buffer[0]+buffer.size();
    this->current_ = (const char*)&buffer[0];
  }

  void copyBuffer(std::vector<char> &outputBuffer){
    outputBuffer = this->buffer;
  }

private:
    std::vector<char> buffer;

    int_type underflow() {
      // fprintf(stderr, "!!! underflow !!!\n");
        if (current_ == end_) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*current_);     // HERE!
    }

    int_type overflow(int_type c){
      // fprintf(stderr, "!!! buffer overflow !!! \n");
      //reallocate
      offset += this->pptr() - begin_;
      buffer.resize(buffer.size()+this->allocationIncrement);
      fprintf(stderr, "offset: %ld  new size: %ld\n", offset, buffer.size());
      // this->begin_ = (const char*)&buffer[0];
      this->end_ = (const char*)&buffer[0]+buffer.size();
      this->current_ = (const char*)&buffer[0] + offset;
      this->begin_ = this->current_;

      this->setp( (char*)begin_, (char*)end_ );
      // pbump(1);
      // this->pubseekpos(offset);
      *pptr() = traits_type::to_char_type(c);
      pbump(1);
      return c;
      // return traits_type::to_int_type(*current_);     //
    }

    int_type uflow() {
        // fprintf(stderr, "!!! uflow !!!\n");
        if (current_ == end_) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*current_++);   // HERE!
    }

    int_type pbackfail(int_type ch) {
        // fprintf(stderr, "!!! pbackfail !!!\n");
        if (current_ == begin_ || (ch != traits_type::eof() && ch != current_[-1])) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*--current_);   // HERE!
    }

    std::streamsize showmanyc() {
        return end_ - current_;
    }

    size_t allocationIncrement;
    size_t offset;
    const char * begin_;
    const char * end_;
    const char * current_;
};


class membuf : public std::streambuf {
public:
  membuf(const char *data, unsigned int len, bool isOutput=true)
  : begin_(data), end_(data + len), current_(data) {
    if(isOutput)
      this->setp( (char*)begin_, (char*)end_ );
  }

  size_t outputCount(){
    return this->pptr() - this->pbase();;
  }

private:
    int_type underflow() {
        if (current_ == end_) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*current_);     // HERE!
    }

    int_type overflow(){
      return traits_type::to_int_type(*current_);     //
    }

    int_type uflow() {
        if (current_ == end_) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*current_++);   // HERE!
    }

    int_type pbackfail(int_type ch) {
        if (current_ == begin_ || (ch != traits_type::eof() && ch != current_[-1])) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*--current_);   // HERE!
    }

    std::streamsize showmanyc() {
        return end_ - current_;
    }

    const char * const begin_;
    const char * const end_;
    const char * current_;
};



#endif // MEMSTREAMBUF_HPP
